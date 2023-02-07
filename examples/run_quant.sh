
for M in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'; do python3 train_inpt_spikingjelly.py --workdir=../../tcja_quant_${M} --config=tcja/configs/quant.py --config.quant.bits=${M}; done