import pandas as pd 

def count_keyword(csv_file, column_name, keyword):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Count instances of the keyword in the specified column
    count = df[column_name].str.contains(keyword, case=False).sum()

    return count


csv_file = 'server/database/dataset/archive/dishes.csv'
column_name = 'name'
keyword = "cake"
instances_count = count_keyword(csv_file, column_name, keyword)

print(f'The keyword "{keyword}" appears {instances_count} times in the column "{column_name}" of the CSV file "{csv_file}"')
