import os
import requests

def download_pdb(pdb_id, save_path):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {pdb_id}')
    else:
        print(f'Failed to download {pdb_id}')

def download_multiple_pdbs(pdb_ids, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for pdb_id in pdb_ids:
        file_path = os.path.join(directory, f'{pdb_id}.pdb')
        download_pdb(pdb_id, file_path)

# Example list of PDB IDs (replace with actual IDs you want to download)
pdb_ids = ['1a3n', '1b3b', '2hbs','1bx7', '1DF4', '2JOX', '1COS', '1DGV', '7UG5', '1GK7', '1NKD', '2OL9', '3MD4', '1DBP', '7UI2', '4AXY', '1NOT', '3Q2X', 'IHJE', '1DK8', '7UJ2', '1HHP', '1A3N', '1J4N', '3UON']  # Add more IDs as needed
pdb_directory = 'pdb_files'
download_multiple_pdbs(pdb_ids, pdb_directory)
