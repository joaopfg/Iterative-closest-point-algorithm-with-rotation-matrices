#include "ICP.h"

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2){
  // return the nearest neighbour to V1 in V2 as nn_V2
  
  // Using octree

	cout << "Computing time using octree..." << endl;
  auto start = std::chrono::high_resolution_clock::now();

	vector<std::vector<int > > O_PI;
	MatrixXi O_CH;
	MatrixXd O_CN;
	VectorXd O_W;
	igl::octree(V2,O_PI,O_CH,O_CN,O_W);

	MatrixXi I;
	igl::knn(V1,2,O_PI,O_CH,O_CN,O_W,I);

	for(int i=0;i<V1.rows();i++){
		nn_V2(i,0) = V2(I(i,1),0);
		nn_V2(i,1) = V2(I(i,1),1);
		nn_V2(i,2) = V2(I(i,1),2);
	}

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Running time for octree: " << elapsed.count() << " s\n";
  

  // Using brute force nearest neighbor (bonus)
/*
  cout << "Computing time using brute force..." << endl;
  auto start = std::chrono::high_resolution_clock::now();

	bool visit[V2.rows()];
	memset(visit, false, sizeof(visit));

	for(int i=0;i<V1.rows();i++){
		MatrixXd pMin(1,3);
		float dMin = -1.0;
		int indMin;

		for(int j=0;j<V2.rows();j++){
			if(!visit[j] && dMin == -1.0 || sqrt((V1(i,0) - V2(j,0))*(V1(i,0) - V2(j,0)) + (V1(i,1) - V2(j,1))*(V1(i,1) - V2(j,1)) + (V1(i,2) - V2(j,2))*(V1(i,2) - V2(j,2))) < dMin){
				dMin = sqrt((V1(i,0) - V2(j,0))*(V1(i,0) - V2(j,0)) + (V1(i,1) - V2(j,1))*(V1(i,1) - V2(j,1)) + (V1(i,2) - V2(j,2))*(V1(i,2) - V2(j,2)));
				pMin(0,0) = V2(j,0);
				pMin(0,1) = V2(j,1);
				pMin(0,2) = V2(j,2);
				indMin = j;
			}
		}

		nn_V2(i,0) = pMin(0,0);
		nn_V2(i,1) = pMin(0,1);
		nn_V2(i,2) = pMin(0,2);
		visit[indMin] = true;
	}
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Running time for brute force: " << elapsed.count() << " s\n";  */
}


void transform(MatrixXd &V1,const MatrixXd &V2){
  //align V1 to V2 when V1 and V2 points are in correspondance
  MatrixXd ux = MatrixXd::Zero(1, 3);
  MatrixXd uy = MatrixXd::Zero(1, 3);

  for(int i=0;i<V1.rows();i++){
    ux(0,0) += V1(i,0);
    ux(0,1) += V1(i,1);
    ux(0,2) += V1(i,2);
  }

  for(int i=0;i<V2.rows();i++){
    uy(0,0) += V2(i,0);
    uy(0,1) += V2(i,1);
    uy(0,2) += V2(i,2);
  } 

  ux(0,0) /= V1.rows();
  ux(0,1) /= V1.rows();
  ux(0,2) /= V1.rows();

  uy(0,0) /= V2.rows();
  uy(0,1) /= V2.rows();
  uy(0,2) /= V2.rows();

  MatrixXd C = MatrixXd::Zero(3,3);

  for(int i=0;i<V1.rows();i++){
    MatrixXd crawlY(3,1), crawlX(1,3);

    crawlY(0,0) = V2(i,0) - uy(0,0);
    crawlY(1,0) = V2(i,1) - uy(0,1);
    crawlY(2,0) = V2(i,2) - uy(0,2);
    crawlX(0,0) = V1(i,0) - ux(0,0);
    crawlX(0,1) = V1(i,1) - ux(0,1);
    crawlX(0,2) = V1(i,2) - ux(0,2);

    C += crawlY*crawlX;
  }

  JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
  MatrixXd U = svd.matrixU();
  MatrixXd V = svd.matrixV();
  float det = (U*V.transpose()).determinant();

  MatrixXd Ropt;

  if(det == 1.0) Ropt = U*V.transpose();
  else{
    MatrixXd s = MatrixXd::Identity(3,3);
    s(2,2) = -1;
    Ropt = U*s;
    Ropt *= V.transpose();
  }

  MatrixXd topt(3,1);
  topt = uy.transpose() - Ropt*ux.transpose();

  for(int i=0;i<V1.rows();i++){
    MatrixXd crawl(3,1);
    crawl(0,0) = V1(i,0);
    crawl(1,0) = V1(i,1);
    crawl(2,0) = V1(i,2);
    crawl = Ropt*crawl;
    crawl += topt;
    V1(i,0) = crawl(0,0);
    V1(i,1) = crawl(1,0);
    V1(i,2) = crawl(2,0);
  }
}
