#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

using namespace Eigen;

std::pair<Matrix3d, Vector3d> randomRT()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0, 0.1);

	// 随机生成一个三维向量并归一化
	Vector3d v(d(gen), d(gen), d(gen));
	v.normalize();

	// 随机生成旋转角度
	double theta = 2 * M_PI * ((double)rand() / RAND_MAX) ;

	// 计算旋转矩阵元素
	double c = cos(theta);
	double s = sin(theta);
	double t = 1 - c;
	double x = v(0), y = v(1), z = v(2);

	Matrix3d R;
	R << t * x * x + c, t * x * y - s * z, t * x * z + s * y,
		t * x * y + s * z, t * y * y + c, t * y * z - s * x,
		t * x * z - s * y, t * y * z + s * x, t * z * z + c;

	// 随机生成平移向量
	Vector3d T(d(gen), d(gen), d(gen));

	return {R, T};
}

std::tuple<double, double, double> cal(int point_num, const MatrixXd& points, const MatrixXd& points_measure,
                                       const Matrix3d& R, const Vector3d& T)
{
	// 质心
	Vector3d p_r = points.colwise().mean();
	Vector3d p_m = points_measure.colwise().mean();

	// 中心化点集
	MatrixXd q_r = points.rowwise() - p_r.transpose();
	MatrixXd q_m = points_measure.rowwise() - p_m.transpose();

	// 协方差矩阵
	Matrix3d H = (q_r.transpose() * q_m).eval();

	// 奇异值分解
	JacobiSVD<MatrixXd> svd(H, ComputeThinU | ComputeThinV);
	Matrix3d U = svd.matrixU();
	Matrix3d V = svd.matrixV();

	// 计算结果
	Matrix3d R_cal = V * U.transpose();
	double det_R = R_cal.determinant();

	Vector3d T_cal = p_m - R_cal * p_r;

	Matrix3d R_diff = R * R_cal.transpose();
	double theta = acos((R_diff.trace() - 1) / 2);
	double diff_R = theta * 180.0 / M_PI; // 弧度转度

	Vector3d trans_diff = T - T_cal;
	double diff_T = trans_diff.norm();

	return {diff_R, diff_T, det_R};
}

int main()
{
	int point_num = 50;
	int for_n = 100000;
	std::vector<double> err_R, err_T;
	std::vector<std::chrono::duration<double>> durations;
	auto t1 = std::chrono::high_resolution_clock::now();
	for (int j = 0; j < for_n; ++j)
	{
		double a = ((double)rand() / RAND_MAX) * 1;
		double b = ((double)rand() / RAND_MAX) * 1;
		double c = ((double)rand() / RAND_MAX) * 1;
		double d = ((double)rand() / RAND_MAX) * 1;

		MatrixXd points = MatrixXd::Random(point_num, 3);
		for (int i = 0; i < point_num; ++i)
		{
			points(i, 2) = (-d - a * points(i, 0) - b * points(i, 1)) / c;
		}

		auto [R, T] = randomRT();
		MatrixXd points_measure = MatrixXd::Zero(point_num, 3);
		for (int i = 0; i < point_num; ++i)
		{
			points_measure.row(i) = R * points.row(i).transpose() + T + MatrixXd::Random(3, 1) * 0.01;
		}

		auto [diff_R, diff_T, det_R] = cal(point_num, points, points_measure, R, T);



		if (diff_R > 90) diff_R -= 90;
		if (det_R > 0.9)
		{
			err_R.push_back(diff_R);
			err_T.push_back(diff_T);
		}
		std::cout <<j<< std::endl;

	}
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Finish " << err_R.size() << " in " << for_n << std::endl;
	std::chrono::duration<double> duration = t2 - t1;

	double mean_R=0, mean_T=0;
	double n = static_cast<double>(err_R.size());
	for (int i = 0; i < n; ++i)
	{
		mean_R += err_R.at(i);
		mean_T += err_T.at(i);
	}
	std::cout << "avg R error: " << mean_R/n << std::endl;
	std::cout << "avg T error: " << mean_T/n << std::endl;
	std::cout << "avg time: " << duration.count()/n << " s"<< std::endl;
}
