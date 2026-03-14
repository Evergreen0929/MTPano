import os

db_root = '/mnt/localssd/MTL'
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]

db_names = {'PASCALContext': 'PASCALContext', 'NYUD_MT': 'NYUDv2'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)


db_paths['Structured3D_MT'] = '/mnt/localssd/Structured3D'
db_paths['Stanford2D3D_MT'] = '/mnt/localssd/Stanford-2D-3D'
db_paths['Matterport3D_MT'] = '/mnt/localssd/Matterport3D_Processed'
db_paths['PanoMTDU'] = '/mnt/localssd/PanoPseudoLabels'
db_paths['SynPASS'] = '/mnt/localssd/SynPASS'
db_paths['Deep360'] = '/mnt/localssd/Deep360_Processed'