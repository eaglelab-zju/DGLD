
for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
do
  expname=$data'_CONAD'
  echo
  echo ${expname}'-----START-----'
  dataset=$data
  python main_conad.py --dataset $dataset --device 1 --logdir logs/$expname > logs/$expname.log 2>&1 &
  cat logs/$expname.log
done
