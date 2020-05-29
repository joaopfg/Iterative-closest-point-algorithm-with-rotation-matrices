#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <bits/stdc++.h>
using namespace Eigen;
using namespace std;
#include <igl/octree.h>
#include <igl/knn.h>

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2);
void transform(MatrixXd &V1,const MatrixXd &V2);
