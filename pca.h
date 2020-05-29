#include <Eigen/Core>
#include <bits/stdc++.h>
using namespace Eigen;
using namespace std;
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <igl/octree.h>
#include <igl/knn.h>

void k_nearest_neighbour(const MatrixXd &V1,Eigen::MatrixXi &I, int k);
void compute_normals(const MatrixXd &V1,const Eigen::MatrixXi &I, int k, MatrixXd &normals);
