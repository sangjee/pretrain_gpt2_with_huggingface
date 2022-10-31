from transformers import GPT2Config, GPT2Tokenizer
from GPT2PretrainModel import GPT2PretrainModel
import pandas as pd
from GPT2PretrainDataModule import GPT2PretrainDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import argparse
import torch

if __name__ == '__main__':
    random_seed = 42
    pl.seed_everything(random_seed, workers=True)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    parser = argparse.ArgumentParser(description='GPT2pretraining')
    parser.add_argument('--model_name', type=str, default='GPT2pretrained')
    parser.add_argument('--save_path', type=str, default='./save_model')
    parser.add_argument('--log_path', type=str, default='./lightning_logs')
    parser.add_argument('--train_data_path', type=str, default='D:/strok_data/dataset/combine(hdf5_nifti)_train.csv')
    parser.add_argument('--test_data_path', type=str, default='D:/strok_data/dataset/combine(hdf5_nifti)_test.csv')
    parser.add_argument('--val_data_path', type=str, default='D:/strok_data/dataset/combine(hdf5_nifti)_val.csv')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--devices', type=int, default=-1)

    args = parser.parse_args()

    config = GPT2Config(
        vocab_size=1255,
        n_positions=512,
        n_ctx=60,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        attn_pdrop=0.1,
        eos_token_id=2,
        bos_token_id=1)


    train_df = pd.read_csv(args.train_data_path)
    test_df = pd.read_csv(args.test_data_path)
    val_df = pd.read_csv(args.val_data_path)

    tokenizer = GPT2Tokenizer(vocab_file='./bbpe/vocab.json', merges_file='./bbpe/merges.txt', eos_token="<|endoftext|>",bos_token="<?", pad_token="+=")
    data_module = GPT2PretrainDataModule(train_df=train_df, val_df=val_df, test_df=test_df, batch_size=args.batch_size, tokenizer=tokenizer, num_workers=args.num_workers)
    model = GPT2PretrainModel(config, lr=args.lr)

    checkpoint_callback = ModelCheckpoint(
         dirpath=args.save_path,
         filename="best-checkpoint",
         save_weights_only=True,
         save_last=True,
         save_top_k=1,
         verbose=True,
         monitor="val_loss",
         mode="min"
         )
    logger = TensorBoardLogger(args.log_path, name=args.model_name)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.epoch,
        accelerator='gpu',
        devices=args.devices
        )

    trainer.fit(model, data_module)