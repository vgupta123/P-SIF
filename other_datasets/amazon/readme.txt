for running with doc2vecc embedding :
python process_amazon_doc2vecc.py
cd doc2vecC
sh go_polysemy_amazon.sh
cd ..
python psif_doc2vecc.py

for running with google embedding:
python process_amazon.py
python curate_embeddings.py
python psif.py
