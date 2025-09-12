import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from datetime import datetime


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    total_epoch = config.epochs
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)

        # print(torch.min(out),torch.max(out))

        # print(out.dtype,targets.dtype)
        # print(out.size(),targets.size())
        loss = criterion(out, targets.detach())

        # print(out.dtype, targets.dtype)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch [{epoch}]/[{total_epoch}], iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}, time:{datetime.now()}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step, np.mean(loss_list)


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):

    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = (f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, '
                    f'\n miou: {miou}, f1_or_dsc: {f1_or_dsc},'
                    f'\n accuracy: {accuracy}, specificity: {specificity}, sensitivity: {sensitivity}, '
                    f'\n confusion_matrix: {confusion}')
        print(log_info)
        logger.info(log_info)

        return np.mean(loss_list), f1_or_dsc, miou, accuracy, specificity, sensitivity

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

        return np.mean(loss_list), np.nan, np.nan, np.nan, np.nan, np.nan


def test_one_epoch(model_name,test_loader, model, criterion, logger, config, test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        MPA = calcuate_MPA(confusion)

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = (
                    f'model_name: {model_name},\n'
                    f'test of best model, loss: {np.mean(loss_list):.4f},\n===================\n'
                    f'Precision: {precision * 100:.2f}  ||'
                    f'Recall: {sensitivity * 100:.2f}\n'
                    f'F1(=dsc): {f1_or_dsc * 100:.2f}  ||'
                    f'Miou: {miou * 100:.2f}\n'
                    f'PA(=Accuracy): {accuracy * 100:.2f} ||'
                    f'MPA: {MPA * 100:.2f} ||'
                    f'Specificity: {specificity * 100:.2f}\n'
                    f'===================confusion_matrix:===================\n{confusion}')
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list), log_info


def calcuate_MPA(confusion_matrix):
    mpa = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    MPA = np.nanmean(mpa)
    return MPA
