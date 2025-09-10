/*
 * Teensy 4.1 距離予測システム (17次多項式モデル)
 * 多項式回帰モデルのデプロイ
 *
 * ハードウェア: Teensy 4.1 (ARM Cortex-M7 @ 600MHz)
 * モデル: 17次多項式（171特徴量, MAE: 0.90cm）
 *
 * 機能:
 * - リアルタイム距離予測
 * - パフォーマンスベンチマーク
 * - 精度検証
 * - メモリ使用量モニタリング
 * - インタラクティブなシリアルインターフェース
 */

#include "teensy_polynomial_model.h"

// 検証用テストデータ構造体
struct TestCase {
    float under_y;
    float theta;
    float expected_distance;
    const char* description;
};

// PC版scikit-learnの正確な結果を使用したテストケース (utils.py経由で生成)
const TestCase TEST_CASES[] = {
    {10.0f, 45.0f, -1.469114477692, "Medium distance, 45 degree angle"},
    {5.0f, 30.0f, 84.307658285533, "Short distance, 30 degree angle"},
    {20.0f, 60.0f, 36887.814311023918, "Long distance, 60 degree angle"},
    {0.0f, 0.0f, 90.490019380227, "Origin point test"},
    {-5.0f, -30.0f, -12.680106555053, "Negative values test"},
    {15.0f, 90.0f, 100166717.672445982695, "90 degree angle test"},
    {8.0f, 15.0f, 75.597363299905, "Low angle test"},
    {25.0f, 120.0f, 11399008744.517791748047, "High angle test"},
    {12.0f, 75.0f, 3874011.306283618789, "High precision test case"},
    {-10.0f, -45.0f, -7951.200055459812, "Negative quadrant test"}
};

const int NUM_TEST_CASES = sizeof(TEST_CASES) / sizeof(TestCase);

// ベンチマーク統計情報
struct BenchmarkStats {
    uint32_t min_time_us;
    uint32_t max_time_us;
    uint32_t total_time_us;
    uint32_t num_samples;
    float mean_error;
    float max_error;
    float total_error;
};

// グローバル変数
BenchmarkStats benchmark_stats;
bool continuous_mode = false;

void setup() {
    // Initialize serial communication
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
        // Wait for serial connection or timeout after 3 seconds
    }
    
    // Display startup information
    Serial.println("\n==================================================");
    Serial.println("Teensy 4.1 距離予測システム (17次多項式モデル)");
    Serial.println("多項式回帰モデルのデプロイ");
    Serial.println("==================================================");
    
    // Print model information
    print_model_info();
    
    // Print memory information
    print_memory_info();
    
    // Initialize benchmark statistics
    reset_benchmark_stats();
    
    // Display menu
    print_menu();
}

void loop() {
    if (Serial.available()) {
        char command = Serial.read();
        handle_command(command);
    }
    
    if (continuous_mode) {
        run_continuous_benchmark();
        delay(100);  // Small delay to prevent overwhelming output
    }
}

void handle_command(char command) {
    switch (command) {
        case '1':
            run_single_prediction();
            break;
        case '2':
            run_test_suite();
            break;
        case '3':
            run_benchmark_suite();
            break;
        case '4':
            toggle_continuous_mode();
            break;
        case '5':
            print_memory_info();
            break;
        case '6':
            reset_benchmark_stats();
            break;
        case '7':
            run_stress_test();
            break;
        case '8':
            run_pc_comparison_test();
            break;
        case 'h':
        case 'H':
            print_menu();
            break;
        case '\n':
        case '\r':
            // Ignore newline characters
            break;
        default:
            Serial.println("無効なコマンドです。「h」を押してヘルプを表示してください。");
            break;
    }
}

void print_menu() {
    Serial.println("\n----------------------------------------");
    Serial.println("コマンドメニュー:");
    Serial.println("1 - 単一予測 (手動入力)");
    Serial.println("2 - テストスイートの実行 (検証)");
    Serial.println("3 - ベンチマークスイートの実行 (パフォーマンス)");
    Serial.println("4 - 連続モードの切り替え");
    Serial.println("5 - メモリ情報の表示");
    Serial.println("6 - ベンチマーク統計のリセット");
    Serial.println("7 - ストレステスト (高負荷)");
    Serial.println("8 - PC版との精度比較テスト (詳細デバッグ付き)");
    Serial.println("h - このメニューを表示");
    Serial.println("----------------------------------------");
    Serial.print("コマンドを入力してください: ");
}

void run_single_prediction() {
    Serial.println("\n=== 単一予測モード ===");
    delay(100);
    // Get input values from user
    float under_y = get_float_input("under_y の値を入力してください (-100 から 100): ");
    float theta = get_float_input("theta の値を入力してください (-180 から 180): ");
    
    // Validate input range
    if (!validate_input_range(under_y, theta)) {
        Serial.println("エラー: 入力値が範囲外です！");
        return;
    }
    
    // Measure prediction time
    uint32_t start_time = micros();
    float prediction = predict_distance_teensy(under_y, theta);
    uint32_t end_time = micros();
    uint32_t execution_time = end_time - start_time;
    
    // Display results
    Serial.println("\n結果:");
    Serial.print("  入力: under_y="); Serial.print(under_y, 3);
    Serial.print(", theta="); Serial.println(theta, 3);
    Serial.print("  予測距離: "); Serial.print(prediction, 3); Serial.println(" cm");
    Serial.print("  実行時間: "); Serial.print(execution_time); Serial.println(" μs");
    
    // Performance analysis for degree 17
    Serial.println("\n17次多項式モデル分析 (Double precision):");
    Serial.print("  特徴量数: "); Serial.println(FEATURE_COUNT);
    Serial.print("  予想実行時間: <120μs");
    Serial.print(" ["); Serial.print(execution_time < 120 ? "合格" : "要最適化"); Serial.println("]");
    
    // Update benchmark statistics
    update_benchmark_stats(execution_time, 0.0f);  // No error calculation for manual input
}

float get_float_input(const char* prompt) {
    Serial.print(prompt);
    while (Serial.available()) {
        Serial.read();
    }
    while (!Serial.available()) {
        // Wait for input
    }
    
    float value = Serial.parseFloat();
    
    // Clear remaining characters in buffer
    while (Serial.available()) {
        Serial.read();
    }
    
    return value;
}

void run_test_suite() {
    Serial.println("\n=== テストスイート (精度検証) ===");
    Serial.println("事前定義されたテストケースを実行中...\n");
    
    float total_error = 0.0f;
    float max_error = 0.0f;
    int passed_tests = 0;
    uint32_t total_time = 0;
    
    for (int i = 0; i < NUM_TEST_CASES; i++) {
        const TestCase& test = TEST_CASES[i];
        
        // Run prediction
        uint32_t start_time = micros();
        float prediction = predict_distance_teensy(test.under_y, test.theta);
        uint32_t end_time = micros();
        uint32_t execution_time = end_time - start_time;
        total_time += execution_time;
        
        // Calculate error
        float error = abs(prediction - test.expected_distance);
        total_error += error;
        if (error > max_error) {
            max_error = error;
        }
        
        // Check if test passes (realistic tolerance for 17th degree polynomial)
        float base_tolerance = 1e-4f;  // Base tolerance for small values
        float relative_tolerance = abs(test.expected_distance) * 1e-4f;  // Relative tolerance
        float tolerance = max(base_tolerance, relative_tolerance);
        bool passed = error <= tolerance;
        if (passed) passed_tests++;
        
        // Display test result
        Serial.print("テスト "); Serial.print(i + 1); Serial.print(": ");
        Serial.print(test.description);
        Serial.println();
        Serial.print("  入力: ("); Serial.print(test.under_y, 1);
        Serial.print(", "); Serial.print(test.theta, 1); Serial.println(")");
        Serial.print("  期待値: "); Serial.print(test.expected_distance, 2);
        Serial.print(" cm, 予測値: "); Serial.print(prediction, 2); Serial.println(" cm");
        Serial.print("  誤差: "); Serial.print(error, 3); Serial.print(" cm");
        Serial.print(", 時間: "); Serial.print(execution_time); Serial.print(" μs");
        Serial.print(" ["); Serial.print(passed ? "合格" : "不合格"); Serial.println("]");
        Serial.println();
        
        // Update benchmark statistics
        update_benchmark_stats(execution_time, error);
    }
    
    // Display summary
    float mean_error = total_error / NUM_TEST_CASES;
    float mean_time = (float)total_time / NUM_TEST_CASES;
    float pass_rate = (float)passed_tests / NUM_TEST_CASES * 100.0f;
    
    Serial.println("=== テストスイート概要 ===");
    Serial.print("合格したテスト: "); Serial.print(passed_tests);
    Serial.print("/"); Serial.print(NUM_TEST_CASES);
    Serial.print(" ("); Serial.print(pass_rate, 1); Serial.println("%)");
    Serial.print("平均誤差: "); Serial.print(mean_error, 3); Serial.println(" cm");
    Serial.print("最大誤差: "); Serial.print(max_error, 3); Serial.println(" cm");
    Serial.print("平均実行時間: "); Serial.print(mean_time, 2); Serial.println(" μs");
    Serial.print("目標: PC版と同等精度 (Double precision)");
    Serial.println();
}

void run_benchmark_suite() {
    Serial.println("\n=== ベンチマークスイート (パフォーマンス測定) ===");
    
    const int BENCHMARK_ITERATIONS = 1000;
    Serial.print(BENCHMARK_ITERATIONS);
    Serial.println(" 回のイテレーションでパフォーマンス分析を実行中...");
    
    uint32_t min_time = UINT32_MAX;
    uint32_t max_time = 0;
    uint32_t total_time = 0;
    
    // Use a representative test case for benchmarking
    float test_under_y = 10.0f;
    float test_theta = 45.0f;
    
    Serial.print("テスト入力: under_y="); Serial.print(test_under_y);
    Serial.print(", theta="); Serial.println(test_theta);
    Serial.println("進行状況: ");
    
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Show progress every 100 iterations
        if (i % 100 == 0) {
            Serial.print(".");
        }
        
        // Measure execution time
        uint32_t start_time = micros();
        float prediction = predict_distance_teensy(test_under_y, test_theta);
        uint32_t end_time = micros();
        uint32_t execution_time = end_time - start_time;
        
        // Update statistics
        total_time += execution_time;
        if (execution_time < min_time) min_time = execution_time;
        if (execution_time > max_time) max_time = execution_time;
        
        // Prevent compiler optimization
        (void)prediction;
    }
    
    Serial.println(" 完了！");
    
    // Calculate statistics
    float mean_time = (float)total_time / BENCHMARK_ITERATIONS;
    
    // Display results
    Serial.println("\n=== ベンチマーク結果 ===");
    Serial.print("イテレーション数: "); Serial.println(BENCHMARK_ITERATIONS);
    Serial.print("最小時間: "); Serial.print(min_time); Serial.println(" μs");
    Serial.print("最大時間: "); Serial.print(max_time); Serial.println(" μs");
    Serial.print("平均時間: "); Serial.print(mean_time, 2); Serial.println(" μs");
    Serial.print("目標時間: <120 μs (Double precision)");
    Serial.print(" ["); Serial.print(mean_time < 120.0f ? "合格" : "要最適化"); Serial.println("]");
    Serial.print("合計時間: "); Serial.print(total_time / 1000.0f, 2); Serial.println(" ms");
    
    // Performance analysis for degree 17
    Serial.println("\n=== パフォーマンス分析 (17次多項式) ===");
    Serial.print("予測あたりのCPUサイクル: ~"); Serial.print((int)(mean_time * 0.6f)); Serial.println(" サイクル");
    Serial.print("理論上の最大予測数/秒: "); Serial.print((int)(1000000.0f / mean_time)); Serial.println();
    Serial.print("特徴量処理効率: "); Serial.print(171.0f / mean_time, 2); Serial.println(" 特徴量/μs");
    
    // Memory efficiency (Double precision)
    Serial.println("\n=== メモリ効率分析 ===");
    Serial.print("スタック使用量: ~"); Serial.print(171 * 8 + 18 * 8 * 2); Serial.println(" bytes (double)");
    Serial.print("フラッシュ使用量: ~"); Serial.print(171 * 8 * 3); Serial.println(" bytes (double)");
    
    // Update global benchmark statistics
    benchmark_stats.min_time_us = min_time;
    benchmark_stats.max_time_us = max_time;
    benchmark_stats.total_time_us += total_time;
    benchmark_stats.num_samples += BENCHMARK_ITERATIONS;
}

void run_stress_test() {
    Serial.println("\n=== ストレステスト (高負荷テスト) ===");
    Serial.println("連続10000回の予測を実行中...");
    
    const int STRESS_ITERATIONS = 10000;
    uint32_t start_total = millis();
    uint32_t failed_predictions = 0;
    float error_sum = 0.0f;
    
    for (int i = 0; i < STRESS_ITERATIONS; i++) {
        // Generate varying test inputs
        float under_y = -50.0f + (i % 100);
        float theta = -90.0f + (i % 180);
        
        // Run prediction
        float prediction = predict_distance_teensy(under_y, theta);
        
        // Check for failed predictions
        if (prediction < 0) {
            failed_predictions++;
        }
        
        // Show progress every 1000 iterations
        if (i % 1000 == 0) {
            Serial.print("進行: "); Serial.print(i); Serial.print("/"); Serial.println(STRESS_ITERATIONS);
        }
    }
    
    uint32_t end_total = millis();
    uint32_t total_duration = end_total - start_total;
    
    Serial.println("\n=== ストレステスト結果 ===");
    Serial.print("総実行時間: "); Serial.print(total_duration); Serial.println(" ms");
    Serial.print("平均予測時間: "); Serial.print((float)total_duration * 1000.0f / STRESS_ITERATIONS, 2); Serial.println(" μs");
    Serial.print("失敗した予測: "); Serial.print(failed_predictions); Serial.print("/"); Serial.println(STRESS_ITERATIONS);
    Serial.print("成功率: "); Serial.print((float)(STRESS_ITERATIONS - failed_predictions) / STRESS_ITERATIONS * 100.0f, 2); Serial.println("%");
    Serial.print("予測レート: "); Serial.print(STRESS_ITERATIONS * 1000.0f / total_duration, 1); Serial.println(" 予測/秒");
}

void run_pc_comparison_test() {
    Serial.println("\n=== PC版との精度比較テスト (詳細デバッグ付き) ===");
    Serial.println("PC版scikit-learnの結果と比較し、内部計算も表示します...\n");
    
    // Test cases with expected PC results
    struct PCTestCase {
        float under_y;
        float theta;
        double expected_pc_result;
        const char* description;
    };
    
    const PCTestCase pc_test_cases[] = {
        {10.0f, 45.0f, -1.469114474846, "Standard test case"},
        {0.0f, 0.0f, 90.490019373702, "Origin point"},
        {-5.0f, -30.0f, -12.680106587601, "Negative values"},
        {15.0f, 90.0f, 100166717.672446146607, "High angle"},
        {5.0f, 30.0f, 84.307658247756, "Low values"}
    };
    
    const int num_pc_tests = sizeof(pc_test_cases) / sizeof(PCTestCase);
    
    float total_error = 0.0f;
    float max_error = 0.0f;
    int passed_tests = 0;
    
    for (int i = 0; i < num_pc_tests; i++) {
        const PCTestCase& test = pc_test_cases[i];
        
        Serial.print("=== テスト "); Serial.print(i + 1); Serial.print(": ");
        Serial.print(test.description); Serial.println(" ===");
        Serial.print("入力: under_y="); Serial.print(test.under_y);
        Serial.print(", theta="); Serial.println(test.theta);
        Serial.print("PC期待結果: "); Serial.println(test.expected_pc_result, 12);
        
#if USE_DOUBLE_PRECISION
        // Detailed step-by-step calculation
        double features[FEATURE_COUNT];
        
        Serial.println("\\n内部計算詳細:");
        Serial.println("ステップ1: 多項式特徴量生成");
        uint32_t step1_start = micros();
        generate_polynomial_features_double(test.under_y, test.theta, features);
        uint32_t step1_time = micros() - step1_start;
        Serial.print("  時間: "); Serial.print(step1_time); Serial.println(" μs");
        Serial.print("  特徴量[0] (bias): "); Serial.println((float)features[0], 8);
        Serial.print("  特徴量[170] (最高次): "); 
        if (abs(features[170]) > 1e30) {
            Serial.println("極大値 (>1e30)");
        } else {
            Serial.println((float)features[170], 8);
        }
        
        Serial.println("ステップ2: 標準化");
        uint32_t step2_start = micros();
        apply_standard_scaling_double(features);
        uint32_t step2_time = micros() - step2_start;
        Serial.print("  時間: "); Serial.print(step2_time); Serial.println(" μs");
        Serial.print("  標準化後[0]: "); Serial.println((float)features[0], 8);
        Serial.print("  標準化後[170]: "); Serial.println((float)features[170], 8);
        
        Serial.println("ステップ3: 線形結合");
        uint32_t step3_start = micros();
        double detailed_result = compute_linear_combination_double(features);
        uint32_t step3_time = micros() - step3_start;
        Serial.print("  時間: "); Serial.print(step3_time); Serial.println(" μs");
        Serial.print("  結果: "); Serial.println((float)detailed_result, 12);
#endif
        
        // Run main prediction function
        uint32_t start_time = micros();
        float teensy_result = predict_distance_teensy(test.under_y, test.theta);
        uint32_t end_time = micros();
        uint32_t execution_time = end_time - start_time;
        
        // Calculate error vs PC result
        double error = abs((double)teensy_result - test.expected_pc_result);
        double relative_error = (error / abs(test.expected_pc_result)) * 100.0;
        
        total_error += (float)error;
        if (error > max_error) {
            max_error = (float)error;
        }
        
        // Check if test passes (realistic tolerance considering numerical limitations)
        double base_tolerance = 1e-5;  // Base tolerance for small values
        double relative_tolerance = abs(test.expected_pc_result) * 1e-5;  // Relative tolerance for large values
        double tolerance = max(base_tolerance, relative_tolerance);
        bool passed = error <= tolerance;
        if (passed) passed_tests++;
        
        // Display results
        Serial.println("\\n最終結果:");
        Serial.print("  Teensy結果: "); Serial.println(teensy_result, 12);
        Serial.print("  絶対誤差: "); Serial.print(error, 12);
        Serial.print("  相対誤差: "); Serial.print(relative_error, 8); Serial.println("%");
        Serial.print("  総実行時間: "); Serial.print(execution_time); Serial.print(" μs");
        Serial.print(" ["); Serial.print(passed ? "合格" : "不合格"); Serial.println("]");
        Serial.println("----------------------------------------\\n");
    }
    
    // Display summary
    float mean_error = total_error / num_pc_tests;
    float pass_rate = (float)passed_tests / num_pc_tests * 100.0f;
    
    Serial.println("=== PC比較テスト概要 ===");
    Serial.print("合格したテスト: "); Serial.print(passed_tests);
    Serial.print("/"); Serial.print(num_pc_tests);
    Serial.print(" ("); Serial.print(pass_rate, 1); Serial.println("%)");
    Serial.print("平均絶対誤差: "); Serial.print(mean_error, 12); Serial.println();
    Serial.print("最大絶対誤差: "); Serial.print(max_error, 12); Serial.println();
    Serial.println("精度評価: " + String(pass_rate >= 80.0f ? "優秀" : (pass_rate >= 60.0f ? "良好" : "要改善")));
    Serial.println("実装: Double precision (64-bit) + Kahan summation");
    Serial.println("注意: 17次多項式の数値特性により、PC版との微小差は正常です");
}



void toggle_continuous_mode() {
    continuous_mode = !continuous_mode;
    Serial.print("連続モード: ");
    Serial.println(continuous_mode ? "有効" : "無効");
    
    if (continuous_mode) {
        Serial.println("連続ベンチマークを実行中... 停止するには任意のキーを押してください。");
        reset_benchmark_stats();
    }
}

void run_continuous_benchmark() {
    static uint32_t last_report_time = 0;
    static int iteration_count = 0;
    
    // Use varying test inputs for more realistic testing
    float test_under_y = 10.0f + sin(millis() * 0.001f) * 15.0f;
    float test_theta = 45.0f + cos(millis() * 0.0015f) * 30.0f;
    
    // Measure execution time
    uint32_t start_time = micros();
    float prediction = predict_distance_teensy(test_under_y, test_theta);
    uint32_t end_time = micros();
    uint32_t execution_time = end_time - start_time;
    
    // Update statistics
    update_benchmark_stats(execution_time, 0.0f);
    iteration_count++;
    
    // Report every 5 seconds
    if (millis() - last_report_time > 5000) {
        Serial.print("連続: "); Serial.print(iteration_count);
        Serial.print(" イテレーション, 最新: "); Serial.print(execution_time);
        Serial.print(" μs, 予測: "); Serial.print(prediction, 2);
        Serial.print(" cm, 平均: "); Serial.print((float)benchmark_stats.total_time_us / benchmark_stats.num_samples, 2);
        Serial.println(" μs");
        
        last_report_time = millis();
        iteration_count = 0;
    }
    
    // Prevent compiler optimization
    (void)prediction;
}

void update_benchmark_stats(uint32_t execution_time, float error) {
    if (benchmark_stats.num_samples == 0) {
        benchmark_stats.min_time_us = execution_time;
        benchmark_stats.max_time_us = execution_time;
    } else {
        if (execution_time < benchmark_stats.min_time_us) {
            benchmark_stats.min_time_us = execution_time;
        }
        if (execution_time > benchmark_stats.max_time_us) {
            benchmark_stats.max_time_us = execution_time;
        }
    }
    
    benchmark_stats.total_time_us += execution_time;
    benchmark_stats.total_error += error;
    benchmark_stats.num_samples++;
    
    if (error > benchmark_stats.max_error) {
        benchmark_stats.max_error = error;
    }
    
    benchmark_stats.mean_error = benchmark_stats.total_error / benchmark_stats.num_samples;
}

void reset_benchmark_stats() {
    benchmark_stats.min_time_us = 0;
    benchmark_stats.max_time_us = 0;
    benchmark_stats.total_time_us = 0;
    benchmark_stats.num_samples = 0;
    benchmark_stats.mean_error = 0.0f;
    benchmark_stats.max_error = 0.0f;
    benchmark_stats.total_error = 0.0f;
    
    Serial.println("ベンチマーク統計をリセットしました。");
}

void print_memory_info() {
    Serial.println("\n=== メモリ情報 (17次多項式モデル) ===");
    
    // Teensy 4.1 memory layout
    Serial.println("Teensy 4.1 メモリレイアウト:");
    Serial.println("  合計RAM: 1024 KB (512KB TCM + 512KB RAM2)");
    Serial.println("  合計フラッシュ: 8192 KB");
    Serial.println("  FPU: ハードウェア浮動小数点演算");
    Serial.println();
    
    // Estimate model memory usage for degree 17
    int model_flash_usage = sizeof(MODEL_COEFFICIENTS) + sizeof(SCALER_MEAN) + sizeof(SCALER_SCALE);
    int feature_array_ram = FEATURE_COUNT * sizeof(float);  // 171 * 4 = 684 bytes
    int power_arrays_ram = 18 * 4 * 2;  // under_y_powers + theta_powers = 144 bytes
    int total_stack_usage = feature_array_ram + power_arrays_ram + 512;  // +512 for other variables
    
    Serial.println("17次多項式モデルメモリ使用量:");
    Serial.print("  モデル係数 (フラッシュ): "); Serial.print(sizeof(MODEL_COEFFICIENTS)); Serial.println(" バイト");
    Serial.print("  スケーラーパラメータ (フラッシュ): "); Serial.print(sizeof(SCALER_MEAN) + sizeof(SCALER_SCALE)); Serial.println(" バイト");
    Serial.print("  合計モデルデータ (フラッシュ): "); Serial.print(model_flash_usage); Serial.println(" バイト");
    Serial.print("  特徴量配列 (double, スタック): "); Serial.print(171 * 8); Serial.println(" バイト");
    Serial.print("  累乗配列 (double, スタック): "); Serial.print(18 * 8 * 2); Serial.println(" バイト");
    Serial.print("  推定スタック使用量 (double精度): "); Serial.print(171 * 8 + 18 * 8 * 2 + 512); Serial.println(" バイト");
    Serial.println();
    
    // Memory efficiency analysis
    float flash_usage_percent = (float)model_flash_usage / (8192 * 1024) * 100.0f;
    float ram_usage_percent = (float)total_stack_usage / (1024 * 1024) * 100.0f;
    
    Serial.println("メモリ効率分析:");
    Serial.print("  フラッシュ使用率: "); Serial.print(flash_usage_percent, 4); Serial.println("%");
    Serial.print("  RAM使用率: "); Serial.print(ram_usage_percent, 4); Serial.println("%");
    Serial.print("  メモリ最適化: "); Serial.println(ram_usage_percent < 0.1f ? "優秀" : "良好");
    
    Serial.println("\n17次多項式の特徴:");
    Serial.print("  特徴量数: "); Serial.println(FEATURE_COUNT);
    Serial.print("  計算複雑度: O(n^2) where n="); Serial.println(POLY_DEGREE);
    Serial.print("  予想実行時間: <100μs");
    Serial.println();
}