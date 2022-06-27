if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo log dir exist
fi

batch=300
for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
do
  expname=$data'_CoLA'
  echo ${expname}
  dataset=$data
  CUDA_VISIBLE_DEVICES=4 python main_cola.py --dataset $dataset --device 0 --batch_size $batch --logdir log/$expname > log/$expname.log 2>&1 &
done
