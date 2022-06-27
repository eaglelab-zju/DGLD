if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo log dir exist
fi

seed=4096
for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
do
  expname=$data'_ComGA'
  echo ${expname}
  dataset=$data
  CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=$seed python main_comga.py --dataset $dataset --device 0 --seed $seed --logdir log/$expname > log/$expname.log 2>&1
done



