#include "pca.h"


void k_nearest_neighbour(const MatrixXd &V1,Eigen::MatrixXi &I, int k){
  MatrixXd P(V1.rows(), 3);

  for(int i=0;i<V1.rows();i++){
    P(i,0) = V1(i,0);
    P(i,1) = V1(i,1);
    P(i,2) = V1(i,2);
  }

  std::vector<std::vector<int > > O_PI;
  Eigen::MatrixXi O_CH;
  Eigen::MatrixXd O_CN;
  Eigen::VectorXd O_W;
  igl::octree(P,O_PI,O_CH,O_CN,O_W);
  igl::knn(P,k+1,O_PI,O_CH,O_CN,O_W,I); 
}


void compute_normals(const MatrixXd &V1,const Eigen::MatrixXi &I, int k, MatrixXd &normals){
	for(int i=0;i<V1.rows();i++){
		MatrixXd I_center = MatrixXd::Zero(1, 3);

		for(int j=1;j<=k;j++){
    		I_center(0,0) += V1(I(i,j),0);
    		I_center(0,1) += V1(I(i,j),1);
    		I_center(0,2) += V1(I(i,j),2);
  		}

  		I_center(0,0) /= k;
    	I_center(0,1) /= k;
    	I_center(0,2) /= k;

    	MatrixXd C = MatrixXd::Zero(3,3);

  		for(int j=1;j<=k;j++){
    		MatrixXd crawl(3,1), crawlT(1,3);

    		crawl(0,0) = V1(I(i,j),0) - I_center(0,0);
    		crawl(1,0) = V1(I(i,j),1) - I_center(0,1);
    		crawl(2,0) = V1(I(i,j),2) - I_center(0,2);
    		crawlT(0,0) = crawl(0,0);
    		crawlT(0,1) = crawl(1,0);
    		crawlT(0,2) = crawl(2,0);

    		C += crawl*crawlT;
  		}

  		C /= k;

  		EigenSolver<MatrixXd> es(C, true);
  		float evalMin = -1.0;
  		int indMin;

  		for(int j=0;j<3;j++){
  			if(evalMin == -1.0 || real((es.eigenvalues())(j,0)) < evalMin){
  				evalMin = real((es.eigenvalues())(j,0));
  				indMin = j;
  			}
  		}

  		normals(i,0) = real((es.eigenvectors().col(indMin))(0,0));
  		normals(i,1) = real((es.eigenvectors().col(indMin))(1,0));
  		normals(i,2) = real((es.eigenvectors().col(indMin))(2,0));
	}
}
