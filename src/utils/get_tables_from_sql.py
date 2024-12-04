import re
import os

def extract_tables_from_sql(sql_query):
    # Regular expression pattern to match table names
    pattern = r'(?i)(?:from|join)\s+([a-zA-Z0-9_\.]+)'
    
    # Find all matches in the SQL query
    matches = re.findall(pattern, sql_query)
    
    # Use a set to avoid duplicate table names
    tables = set(matches)
    
    return tables

def read_sql_from_file(file_name):
    with open(file_name, 'r') as file:
        sql_query = file.read()
    return sql_query

def main():
    # Specify the name of your SQL file (ensure it's in the same directory)
    sql_file_name = 'sql.txt'  # Change this to your actual filename
    
    # Check if the file exists
    if os.path.exists(sql_file_name):
        # Read SQL query from file
        sql_query = read_sql_from_file(sql_file_name)
        
        # Extract tables from SQL query
        tables = extract_tables_from_sql(sql_query)
        
        print("Tables found in SQL query:")
        for table in tables:
            print(table)
    else:
        print(f"File '{sql_file_name}' not found.")

if __name__ == "__main__":
    main()