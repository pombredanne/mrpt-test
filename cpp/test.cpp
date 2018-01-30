#include <iostream>
#include "cpp/Mrpt.h"
#include <Eigen/Dense>
#include <random>
#include <omp.h>
#include <vector>
#include <cmath>

using Eigen::MatrixXf;
using Eigen::Map;
using Eigen::VectorXf;

MatrixXf generate_random_matrix(int nrow, int ncol) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> uni_dist(0, 1);

  MatrixXf m(nrow,ncol);
  for(int i = 0; i < nrow; ++i)
    for(int j = 0; j < ncol; ++j)
      m(i,j) = uni_dist(gen);

  return m;
}

MatrixXf generate_data(int n, int dim, int ncol = 5) {
  MatrixXf a = generate_random_matrix(dim, ncol);
  MatrixXf b = generate_random_matrix(ncol, n);
  return a * b;
}

double mean(const std::vector<double> &x) {
  int n = x.size();
  double sum = 0;
  for(int i = 0; i < n; ++i) sum += x[i];
  return sum / n;
}

double sum(const std::vector<double> &x) {
  int n = x.size();
  double sum = 0;
  for(int i = 0; i < n; ++i) sum += x[i];
  return sum;
}

int correct_out_of_k(const std::vector<int> &correct, const std::vector<int> &approximate) {
  int k = correct.size();
  int k2 = approximate.size();
  int n_correct = 0;
  for(int i = 0; i < k2; ++i)
    for(int j = 0; j < k; ++j)
      if(approximate[i] == correct[j]) {
        n_correct++;
      }

  return n_correct;
}

int main(int argc, char** argv) {
  int k = 10;
  int n_queries = 100;
  int ncol = 5;
  int dim = 100;
  int n_points = 100000;

  int n_trees = 100;
  int depth = 5;
  int votes_required = 4;
  float sparsity = 1.0 / std::sqrt(dim);


  MatrixXf data = generate_data(n_points, dim);
  MatrixXf queries = generate_data(n_queries, dim);
  // Map<MatrixXf> data2(data.data(), dim, n);
  const Map<const MatrixXf> *data2 = new Map<const MatrixXf>(data.data(), dim, n_points);

  std::cout << std::endl << data2->block(0,0,5,5) << std::endl;
  std::cout << "data dim: " << data2->rows() << std::endl;
  std::cout << "data sample size: " << data2->cols() << std::endl << std::endl;

  Mrpt index(data2, n_trees, depth, sparsity);
  index.grow();

  VectorXi idx(n_points);
  std::iota(idx.data(), idx.data() + n_points, 0);

  float *test = queries.data();

  std::vector<double> times_exact(n_queries), times_approximate(n_queries);
  int n_correct = 0;

  omp_set_num_threads(1);
  for (int i = 0; i < n_queries; ++i) {
      std::vector<int> result(k), result_approximate(k);
      std::vector<float> distances(k), distances_approximate(k);

      double start = omp_get_wtime();
      index.exact_knn(Map<VectorXf>(&test[i * dim], dim), k, idx, n_points, &result[0], &distances[0]);
      double end = omp_get_wtime();
      times_exact[i] = end - start;

      start = omp_get_wtime();
      index.query(Map<VectorXf>(&test[i * dim], dim), k, votes_required, &result_approximate[0], &distances_approximate[0]);
      end = omp_get_wtime();
      times_approximate[i] = end - start;

      n_correct += correct_out_of_k(result, result_approximate);

      printf("%g\n", times_exact[i]);
      printf("%g\n", times_approximate[i]);
      for (int j = 0; j < k; ++j) printf("%d ", result[j]);
      printf("\n");
      for (int j = 0; j < k; ++j) printf("%d ", result_approximate[j]);
      printf("\n");
      for (int j = 0; j < k; ++j) printf("%g ", distances[j]);
      printf("\n");
      for (int j = 0; j < k; ++j) printf("%g ", distances_approximate[j]);
      printf("\n");
  }

  std::cout << "\nExact search time for " << n_queries << " queries: " << sum(times_exact) << "\n";
  std::cout << "Approximate search time for " << n_queries << " queries: " << sum(times_approximate) << "\n";
  std::cout << "Number of trees: " << n_trees << ", depth: " << depth << ", vote threshold: " << votes_required << ", sparsity: " << sparsity << "\n";
  std::cout << "Accuracy: " << (1.0 * n_correct) / (n_queries * k) << "\n";

  return 0;
}
