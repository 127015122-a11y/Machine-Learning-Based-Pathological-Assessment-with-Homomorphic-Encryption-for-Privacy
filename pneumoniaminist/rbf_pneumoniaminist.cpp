#define PROFILE

#include "openfhe.h"

using namespace lbcrypto;

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace std;
// #define VERBOSE

// utility functions
void print_double_vector_comma_separated(const vector<double>& data, const string& label) {
    cout << label << ": [";
    for (size_t i = 0; i < data.size() - 1; i++) {
        cout << setprecision(15) << data[i] << ", ";
    }
    cout << setprecision(15) << data[data.size() - 1] << "]" << endl;
}

// utility function to read a vector of doubles
vector<double> read_double_data_from_file(const string& filename) {
    vector<double> data;
    ifstream in(filename);
    if (!in) {
        cerr << "Could not open file " << filename << endl;
        return data;
    }

    double number;
    while (in >> number) {
        data.push_back(number);
    }

    in.close();
    return data;
}

void resize_double_vector(vector<double>& data, uint32_t new_size) {
    if (new_size < data.size()) {
        data.resize(new_size);
    } else {
        for (size_t i = data.size(); i < new_size; i++) {
            data.push_back(0.0);
        }
    }
}

// utility function read a matrix of doubles
vector<vector<double>> read_2d_matrix_from_file(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file " << filename << endl;
        return {};
    }

    vector<vector<double>> matrix;
    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    file.close();
    return matrix;
}

void print_matrix(const vector<vector<double>>& matrix, const string& label) {
    cout << label << ": [\n";
    for (size_t i = 0; i < matrix.size(); i++) {
        cout << "[";
        for (size_t j = 0; j < matrix[i].size(); j++) {
            cout << setw(15) << setprecision(15) << matrix[i][j] << " ";
        }
        cout << "]" << endl;
    }
    cout << "]" << endl;
}

int main() {
    cout << "SVM RBF Kernel v1 started ... !\n\n";

    uint32_t n = 128; // SVM vectors dimensions (# of predictors)
   
    // RBF kernel parameters
    double gamma = 0.05;

    vector<vector<double>> support_vectors = read_2d_matrix_from_file("E:\\rbf_image_pneumonia\\support_vectors_rbf.txt");
    cout << "number of support vectors: " << support_vectors.size() << "\n";
    cout << "dimension of each support vector: " << support_vectors[0].size() << "\n";
    print_matrix(support_vectors, "support vectors");

    // read dual coefficients and bias
    vector<double> dual_coeffs = read_double_data_from_file("E:\\rbf_image_pneumonia\\dual_coeff_rbf.txt");
    vector<double> b_vec = read_double_data_from_file("E:\\rbf_image_pneumonia\\intercept_rbf.txt");
    double bias = b_vec[0]; // corrected

    vector<double> bias_vec(n, 0.0);
    bias_vec[0] = bias;

    // read input test vector
    vector<double> x = read_double_data_from_file("E:\\rbf_image_pneumonia\\xtest_rbf.txt");
    vector<double> y_ground_truth = read_double_data_from_file("E:\\rbf_image_pneumonia\\ytest_rbf.txt");
    resize_double_vector(y_ground_truth, n);
    vector<double> y_expected_score = read_double_data_from_file("E:\\rbf_image_pneumonia\\yclassificationscore.txt");
    resize_double_vector(y_expected_score, n);

    print_double_vector_comma_separated(dual_coeffs, "dual_coeff");
    print_double_vector_comma_separated(bias_vec, "bias");
    print_double_vector_comma_separated(x, "x");
    print_double_vector_comma_separated(y_ground_truth, "y_ground_truth");
    print_double_vector_comma_separated(y_expected_score, "y_expected_score");

    // Step 1: Setup CryptoContext
    uint32_t multDepth = 12;
    uint32_t scaleModSize = 55;
    uint32_t batchSize = n;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << endl << endl;

    // Step 2: Key Generation
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

    // Step 3: Encoding and encryption of inputs
    vector<double> zeros(n, 0.0);
    Plaintext pt_zeros = cc->MakeCKKSPackedPlaintext(zeros);

    vector<double> gamma_vec(n, 0.0);
    gamma_vec[0] = gamma;
    Plaintext pt_gamma = cc->MakeCKKSPackedPlaintext(gamma_vec);

    Plaintext pt_x = cc->MakeCKKSPackedPlaintext(x);
    vector<Plaintext> pt_support_vectors;
    for (auto& vec : support_vectors) {
        pt_support_vectors.push_back(cc->MakeCKKSPackedPlaintext(vec));
    }
    Plaintext pt_bias = cc->MakeCKKSPackedPlaintext(bias_vec);

    auto ct_x = cc->Encrypt(keys.publicKey, pt_x);
    cout << "num levels in input ctxt: " << ct_x->GetLevel() << "\n";
    cout << "num towers in input ctxt: " << ct_x->GetElements()[0].GetAllElements().size() << endl;

    double lowerBound = -100, upperBound = 0.0;
    uint32_t polyDegree = 27;

    // Step 4: Evaluation
    TimeVar t;
    TIC(t);

    vector<double> mask(n, 0.0);
    mask[0] = 1.0;
    Plaintext pt_mask = cc->MakeCKKSPackedPlaintext(mask);

    auto ct_res = cc->Encrypt(keys.publicKey, pt_zeros);

    for (size_t i = 0; i < pt_support_vectors.size(); i++) {
        cout << "iteration: " << i + 1 << "\n";

        auto diff = cc->EvalSub(ct_x, pt_support_vectors[i]);
        diff = cc->EvalSquare(diff);
        diff = cc->EvalMult(diff, -gamma);
        auto sum = cc->EvalSum(diff, batchSize);
        sum = cc->EvalMult(sum, pt_mask);

        auto ct_out = cc->EvalChebyshevFunction(
            [](double y) -> double { return std::exp(y); },
            sum, lowerBound, upperBound, polyDegree
        );

        ct_out = cc->EvalMult(ct_out, dual_coeffs[i]);
        ct_res = cc->EvalAdd(ct_res, ct_out);
    }

    ct_res = cc->EvalAdd(ct_res, pt_bias);

    auto timeEvalSVMTime = TOC_MS(t);
    cout << "RBF-SVM inference took: " << timeEvalSVMTime << " ms\n\n";

    // Step 5: Decryption and output
    Plaintext result;
    cc->Decrypt(keys.secretKey, ct_res, &result);
    result->SetLength(batchSize);

    cout.precision(8);
    cout << endl << "Results of homomorphic computations: " << endl;
    cout << "num levels in result ctxt: " << ct_res->GetLevel() << "\n";
    cout << "num towers in result ctxt: " << ct_res->GetElements()[0].GetAllElements().size() << endl;
    cout << "computed classification score = " << result;
    cout << "Estimated precision in bits: " << result->GetLogPrecision() << endl;

    vector<double> resultVector = result->GetRealPackedValue();
    double score = resultVector[0];
    cout << "decision value: " << score << endl;

    if (score >= 0) {
        cout << "Person is Diseased" << endl;
    } else {
        cout << "Person is not Diseased" << endl;
    }

    cout << "SVM RBF Kernel terminated gracefully ... !" << endl;
    return 0;
}