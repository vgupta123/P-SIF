for running with doc2vecc embedding :
python process_bbc.py
cd doc2vecC
sh go_polysemy_20news_polysemy.sh
cd ..
python psif_doc2vecc.py

for running with google embedding:
python process_bbc.py
python curate_embeddings.py
python psif.py


