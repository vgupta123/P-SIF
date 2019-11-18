for running with doc2vecc embedding :
python process_classic.py
cd doc2vecC
sh go_polysemy_classic.sh
cd ..
python psif_doc2vecc.py

for running with google embedding:
python process_classic_doc2vecc.py
python curate_embeddings.py
python psif.py


