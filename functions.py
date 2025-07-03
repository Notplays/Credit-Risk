import matplotlib.pyplot as plt
import numpy as np
def plot_loan_status_outliers(data, col, status_col='loan_status'):
    """
    Detect outliers in the age column of the dataframe using IQR,
    then plot the loan status distribution among those outliers.

    Parameters:
    - data: pandas DataFrame containing the data
    - col: str, name of the column with ages (default 'person_age')
    - status_col: str, name of the column with loan status (default 'loan_status')
    """
    # Calculate Q1 and Q3
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find outliers
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

    print(f"Upper bound: {upper_bound}")
    print(f"Number of outliers: {len(outliers)}")

    # Calculate loan status percentages within outliers
    loan_status_pct = outliers[status_col].value_counts(normalize=True)

    # Plot pie chart
    plt.figure(figsize=(6,6))
    loan_status_pct.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Loan Status Distribution in the Outliers', fontsize=14, fontweight='bold')
    plt.ylabel('')  # Hide y-label for a cleaner look
    plt.show()





