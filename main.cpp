#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <random>
#include <cmath>

using namespace std;
using namespace cv;
using namespace Eigen;

// 计算点集的质心
array<float, 3> centroid(const vector<array<float, 3>>& points)
{
    array<float, 3> sum = {0.0f, 0.0f, 0.0f}; // 初始化质心为零向量
    for (const auto& point : points)
    {
        // 遍历每个点
        sum[0] += point[0];
        sum[1] += point[1];
        sum[2] += point[2];
    }
    float size = static_cast<float>(points.size());
    return {sum[0] / size, sum[1] / size, sum[2] / size}; // 返回平均值作为质心
}

// Kabsch算法核心部分：计算旋转矩阵
Matrix3d kabschRotation(const vector<array<float, 3>>& A, const vector<array<float, 3>>& B)
{
    // 检查输入是否一致
    if (A.size() != B.size())
    {
        throw runtime_error("Point sets must have the same number of points");
    }

    int n = A.size();

    MatrixXd H = MatrixXd::Zero(3, 3);
    for (int i = 0; i < n; ++i)
    {
        Vector3d a(A[i][0], A[i][1], A[i][2]);
        Vector3d b(B[i][0], B[i][1], B[i][2]);
        H += a * b.transpose();
    }

    JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);

    Matrix3d Vt = svd.matrixV().transpose();
    Matrix3d R = Vt * svd.matrixU().transpose();

    // 确保R是一个合法的旋转矩阵（det(R) = 1）
    double d = (svd.matrixV().determinant() * svd.matrixU().determinant()) < 0 ? -1 : 1;
    Matrix3d U_star = svd.matrixU();
    U_star.col(2) *= d;
    R = Vt * U_star.transpose();

    return R;
}

// 将点集投影到平面上
vector<array<float, 3>> projectToPlane(const vector<array<float, 3>>& points, const Vector3d& normal)
{
    vector<array<float, 3>> projectedPoints;
    for (const auto& point : points)
    {
        Vector3d p(point[0], point[1], point[2]);
        Vector3d projection = p - (p.dot(normal) / normal.squaredNorm()) * normal;
        projectedPoints.push_back({
            static_cast<float>(projection.x()),
            static_cast<float>(projection.y()),
            static_cast<float>(projection.z())
        });
    }
    return projectedPoints;
}

void Kabsch(vector<array<float, 3>> points)
{
    array<float, 3> center = centroid(points);

    // 移动点集使其质心位于原点
    vector<array<float, 3>> centeredPoints;
    for (const auto& point : points)
    {
        centeredPoints.push_back({point[0] - center[0], point[1] - center[1], point[2] - center[2]});
    }

    // 创建一个目标点集，这里假设所有点都在xy平面上
    vector<array<float, 3>> targetPoints;
    for (const auto& point : centeredPoints)
    {
        targetPoints.push_back({point[0], point[1], 0.0f});
    }

    // 使用Kabsch算法计算旋转矩阵
    Matrix3d rotationMatrix = kabschRotation(centeredPoints, targetPoints);

    // 应用旋转矩阵到原始点集上
    vector<array<float, 3>> rotatedPoints;
    for (const auto& point : centeredPoints)
    {
        Vector3d p(point[0], point[1], point[2]);
        Vector3d rotatedPoint = rotationMatrix * p;
        rotatedPoints.push_back({
            static_cast<float>(rotatedPoint.x()),
            static_cast<float>(rotatedPoint.y()),
            static_cast<float>(rotatedPoint.z())
        });
    }

    // 获取旋转后的点集的法向量（即z轴方向）
    Vector3d planeNormal = rotationMatrix.col(2);

    // 将原始点集投影到平面上
    vector<array<float, 3>> projectedPoints = projectToPlane(rotatedPoints, planeNormal);

    // 输出结果
    cout << "Original Points:" << endl;
    for (const auto& point : points)
    {
        cout << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")" << endl;
    }

    cout << "\nProjected Points on Plane:" << endl;
    for (const auto& point : projectedPoints)
    {
        cout << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")" << endl;
    }

    // 输出拟合平面的解析式
    // 平面方程 ax + by + cz + d = 0
    float a = static_cast<float>(planeNormal.x());
    float b = static_cast<float>(planeNormal.y());
    float c = static_cast<float>(planeNormal.z());
    float d = -(a * center[0] + b * center[1] + c * center[2]);

    cout << "\nFitted Plane Equation: " << a << "x + " << b << "y + " << c << "z + " << d << " = 0" << endl;
}

// Function to generate random numbers following a normal distribution
float generateNormalDistribution(float mean, float stddev)
{
    static std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, stddev);
    return distribution(generator);
}

// Function to generate a random plane and points around it
std::pair<std::array<float, 4>, std::vector<std::array<float, 3>>> generateRandomPlaneAndPoints()
{
    // Randomly generate coefficients for the plane equation: ax + by + cz = d
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    float a = dis(gen);
    float b = dis(gen);
    float c = dis(gen);
    float d = dis(gen);

    std::array<float, 4> planeCoefficients = {a, b, c, d};

    // Generate 36 points around the plane with some noise
    std::vector<std::array<float, 3>> points;
    for (int i = 0; i < 36; ++i)
    {
        float x = generateNormalDistribution(0, 1);
        float y = generateNormalDistribution(0, 1);
        float z = (d - a * x - b * y) / c; // Calculate z based on the plane equation

        // Add some noise to the point
        float noiseX = generateNormalDistribution(0, 0.5);
        float noiseY = generateNormalDistribution(0, 0.5);
        float noiseZ = generateNormalDistribution(0, 0.5);

        points.push_back({x + noiseX, y + noiseY, z + noiseZ});
    }

    return {planeCoefficients, points};
}

int main()
{
    auto random_data = generateRandomPlaneAndPoints();
    auto points = random_data.second;
    auto start = std::chrono::high_resolution_clock::now();
    Kabsch(points);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto xyz = random_data.first;
    printf("Real Plane Equation：%f x + %f y + %f z + %f  = 0 \n ",xyz.at(0),xyz.at(1),xyz.at(2),xyz.at(3));
    std::cout << "Function took " << duration.count() << " milliseconds to execute." << std::endl;

    return 0;
}



