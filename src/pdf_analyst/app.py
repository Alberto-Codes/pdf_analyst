import csv

def print_year_and_url(csv_file_path: str):
    """Prints the year and URL from each row in the CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            year = row.get('year')
            url = row.get('url')
            print(f"Year: {year}, URL: {url}")

# Example usage
csv_file_path = 'seeds/wfc_10k.csv'
print_year_and_url(csv_file_path)
