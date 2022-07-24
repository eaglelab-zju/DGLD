
batch=300
for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
do
  expname=$data'_CoLA'
  echo
  echo ${expname}'-----START-----'
  dataset=$data
  python main_cola.py --dataset $dataset --device 0 --batch_size $batch --logdir logs/$expname > logs/$expname.log 2>&1 &
  cat logs/$expname.log
done
