import os
import pandas as pd
import numpy as np

def load_dataset_safely():
    # Target the exact filename present in your working directory
    target_file = "Dataset for Data Analytics.csv"
    
    if os.path.exists(target_file):
        print(f"-> Successfully located dataset: {target_file}")
        df = pd.read_csv(target_file)
        # Intern Requirement: Handle the missing entries before calculating stats
        if 'CouponCode' in df.columns:
            df['CouponCode'] = df['CouponCode'].fillna('NONE')
        return df
    else:
        raise FileNotFoundError(f"Could not find '{target_file}' in this folder. Please double-check your workspace!")

def run_exploratory_data_analysis():
    print("=== STARTING PROJECT 2: EXPLORATORY DATA ANALYSIS ===")
    
    try:
        # 1. Load data safely
        df = load_dataset_safely()
        print(f" Loaded Dataset Shape: {df.shape}\n")
        
        # 2. Calculate Basic Descriptive Statistics (Requirement 1)
        print("--- 1. Basic Descriptive Statistics ---")
        numeric_summary = df[['Quantity', 'UnitPrice', 'TotalPrice', 'ItemsInCart']].describe()
        print(numeric_summary.round(2))
        print("\n")
        
        # 3. Categorical Trends: Revenue by Product (Requirement 2)
        print("--- 2. Performance Trends by Product Category ---")
        product_trends = df.groupby('Product').agg(
            Total_Revenue=('TotalPrice', 'sum'),
            Average_Order_Value=('TotalPrice', 'mean'),
            Total_Units_Sold=('Quantity', 'sum'),
            Transaction_Count=('OrderID', 'count')
        ).sort_values(by='Total_Revenue', ascending=False)
        print(product_trends.round(2))
        print("\n")
        
        # 4. Categorical Trends: Revenue by Payment Method
        print("--- 3. Transaction Breakdown by Payment Method ---")
        payment_trends = df.groupby('PaymentMethod').agg(
            Total_Revenue=('TotalPrice', 'sum'),
            Transaction_Count=('OrderID', 'count')
        ).sort_values(by='Total_Revenue', ascending=False)
        print(payment_trends.round(2))
        print("\n")

        # 5. Outlier Detection: Identify High-Value Orders (Requirement 2)
        print("--- 4. Outliers and High-Value Transaction Profiling ---")
        revenue_threshold = df['TotalPrice'].quantile(0.95)
        outliers = df[df['TotalPrice'] > revenue_threshold]
        print(f"-> Outlier Threshold (95th Percentile): ${revenue_threshold:.2f}")
        print(f"-> Isolated {len(outliers)} high-value outlier transactions out of {len(df)} total orders.")
        print("\nTop 5 High-Value Outlier Samples:")
        print(outliers[['OrderID', 'Product', 'Quantity', 'TotalPrice', 'ReferralSource']].head().to_string(index=False))
        print("\n")
        
        # 6. Save data logs as CSV artifacts for documentation
        product_trends.to_csv("product_performance_summary.csv")
        outliers.to_csv("high_value_outliers.csv", index=False)
        print("[SUCCESS] EDA summary logs exported successfully to folder.")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline run failed: {str(e)}")

if __name__ == "__main__":
    run_exploratory_data_analysis()