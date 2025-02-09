import pandas as pd


def remove_non_consecutive_rows(input_csv, output_csv):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Convert 'ID' column to numeric (to handle non-integer values)
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')

    # Drop rows where 'ID' column is NaN (non-integer values)
    df.dropna(subset=['ID'], inplace=True)

    # Check consecutive indices
    consecutive_indices = df['ID'].diff() == 1

    # Filter DataFrame to include only consecutive rows
    df_consecutive = df[consecutive_indices]

    # Remove rows with missing values in the 'type' column
    df_consecutive = df_consecutive.dropna(subset=['type'])

    # Write DataFrame to output CSV file
    df_consecutive.to_csv(output_csv, index=False)


input_csv_file = 'noisy_data.csv'
output_csv_file = 'dataset.csv'

remove_non_consecutive_rows(input_csv_file, output_csv_file)
