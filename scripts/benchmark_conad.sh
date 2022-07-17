if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo log dir exist
fi


for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
do
  expname=$data'_CONAD'
  echo ${expname}
  dataset=$data
  CUDA_VISIBLE_DEVICES=0,1 python main_conad.py --dataset $dataset --device 1 --logdir log/$expname > log/$expname.log 2>&1 &
done
