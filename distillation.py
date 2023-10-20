import argparse
import os
import json
import random
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader
# from simclr import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import sys

sys.path.append('./')
# import simclr
# import simclr.utils.py
from utils import distribute_over_GPUs, validate_arguments
from model import Model, Identity
from get_dataloader import get_dataloader
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from helpers import get_device, rotate_img, one_hot_embedding

torch.backends.cudnn.benchmark = True
from sklearn.metrics import f1_score


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        # Load pre-trained model
        base_model = Model(pretrained=opt.pretrained)
        # if (not opt.random) and (not opt.pretrained):
        #    print('Loading model from ', opt.pre_model_path)
        #     base_model.load_state_dict(torch.load(opt.pre_model_path)['model'], strict=True)

        self.f = base_model.f
        self.g1 = base_model.g1
        # classifier
        self.fc = nn.Linear(opt.output_dims, opt.num_classes, bias=True)
        # self.fc = base_model.fc

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g1(feature)
        out = self.fc(feature)
        return out


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


num_classes = 9


# train or test for one epoch
def train_val(net, net2, data_loader, train_optimizer, device):
    is_train = train_optimizer is not None
    # net2.eval() # train only the last layers.
    net2.train() if is_train else net2.eval()
    net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)

    all_preds, all_labels, all_slides, all_outputs0, all_outputs1, all_patches = [], [], [], [], [], []
    all_outputs2, all_outputs3, all_outputs4, all_outputs5, all_outputs6, all_outputs7, all_outputs8 = [], [], [], [], [], [], []
    all_evidence = []
    all_evidence = torch.FloatTensor(all_evidence)
    all_evidence = all_evidence.cuda(non_blocking=True)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, _, target, patch_id, slide_id in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            # out = net(data)
            teacher_logits_t = net(data)
            evidence_teach = relu_evidence(teacher_logits_t)
            alpha_teach = evidence_teach + 1
            S_teach = torch.sum(alpha_teach, dim=1, keepdim=True)
            # m = nn.Sigmoid()
            # nirho_coeff=m((1-(num_classes/S_teach)))
            nirho_coeff = (1 - (num_classes / S_teach))
            # nirho_coeff = nirho_coeff.to(device)
            student_logits_t = net2(data)
            _, teacher_labels = torch.max(teacher_logits_t.data, 1)
            if is_train:
                ground_labels = teacher_labels
            else:
                ground_labels = target
            y = one_hot_embedding(teacher_labels, num_classes)
            loss_t = criterion(student_logits_t, (y.float().to(device)), epoch, num_classes, 10, nirho_coeff, device)
            # loss_t = add_kd_loss(student_logits_t, teacher_logits_t, temperature)
            target_weight = None
            # loss_t = loss_criterion3(student_logits_t, teacher_logits_t)

            if is_train:
                train_optimizer.zero_grad()
                loss_t.backward()
                train_optimizer.step()

            _, preds = torch.max(student_logits_t.data, 1)
            match = torch.reshape(torch.eq(preds, teacher_labels).float(), (-1, 1))
            acc = torch.mean(match)
            evidence = relu_evidence(student_logits_t)

            all_evidence = torch.cat((all_evidence, evidence), dim=0)
            total_evidence = torch.sum(evidence, 1, keepdim=True)
            mean_evidence = torch.mean(total_evidence)
            mean_evidence_succ = torch.sum(torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
            mean_evidence_fail = torch.sum(torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (
                        torch.sum(torch.abs(1 - match)) + 1e-20)

            all_preds.extend(preds.cpu().numpy())

            all_labels.extend(ground_labels.cpu().numpy())
            all_patches.extend(patch_id)
            all_slides.extend(slide_id)

            probs = torch.nn.functional.softmax(student_logits_t.data, dim=1).cpu().numpy()
            all_outputs0.extend(probs[:, 0])
            all_outputs1.extend(probs[:, 1])
            all_outputs2.extend(probs[:, 2])
            all_outputs3.extend(probs[:, 3])
            all_outputs4.extend(probs[:, 4])
            all_outputs5.extend(probs[:, 5])
            all_outputs6.extend(probs[:, 6])
            all_outputs7.extend(probs[:, 7])
            all_outputs8.extend(probs[:, 8])

            total_num += data.size(0)
            total_loss += loss_t.item() * data.size(0)
            prediction = torch.argsort(student_logits_t, dim=-1, descending=True)
            teacher_prediction = torch.argsort(teacher_logits_t, dim=-1, descending=True)
            # print(prediction.shape)
            # print(teacher_prediction .shape)
            # total_correct += torch.sum((prediction[:, 0:1] == ground_labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct += torch.sum(
                (prediction[:, 0:1] == teacher_prediction[:, 0:1]).any(dim=-1).float()).item()
            F1 = f1_score(all_labels, all_preds, average='weighted')

            data_bar.set_description(
                f'{"Train" if is_train else "Test"} Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f} ACC: {total_correct / total_num * 100:.2f}% f1: {F1 * 100:.2f}%')

    alpha = all_evidence + 1
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    u = u.cpu().data.numpy()
    prob = prob.cpu().data.numpy()
    u2 = u[:, 0]
    prob_0 = prob[:, 0]
    prob_1 = prob[:, 1]
    prob_2 = prob[:, 2]
    prob_3 = prob[:, 3]
    prob_4 = prob[:, 4]
    prob_5 = prob[:, 5]
    prob_6 = prob[:, 6]
    prob_7 = prob[:, 7]
    prob_8 = prob[:, 8]

    df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_preds,
        'slide_id': all_slides,
        'patch_id': all_patches,
        'probabilities_0': all_outputs0,
        'probabilities_1': all_outputs1,
        'probabilities_2': all_outputs2,
        'probabilities_3': all_outputs3,
        'probabilities_4': all_outputs4,
        'probabilities_5': all_outputs5,
        'probabilities_6': all_outputs6,
        'probabilities_7': all_outputs7,
        'probabilities_8': all_outputs8,
        'uncertainty': u2,
        'prob_0': prob_0,
        'prob_1': prob_1,
        'prob_2': prob_2,
        'prob_3': prob_3,
        'prob_4': prob_4,
        'prob_5': prob_5,
        'prob_6': prob_6,
        'prob_7': prob_7,
        'prob_8': prob_8
    })
    return total_loss / total_num, total_correct / total_num * 100, df, F1


# train or test for one epoch
def test_val(net2, data_loader, train_optimizer, device):
    is_train = train_optimizer is not None
    net2.eval()  # train only the last layers.
    # net2.train() if is_train else net.eval()

    total_loss, total_correct, total_num, data_bar = 0.0, 0.0, 0, tqdm(data_loader)

    all_preds, all_labels, all_slides, all_outputs0, all_outputs1, all_patches = [], [], [], [], [], []
    all_outputs2, all_outputs3, all_outputs4, all_outputs5, all_outputs6, all_outputs7, all_outputs8 = [], [], [], [], [], [], []
    all_evidence = []
    all_evidence = torch.FloatTensor(all_evidence)
    all_evidence = all_evidence.cuda(non_blocking=True)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, _, target, patch_id, slide_id in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            student_logits_t = net2(data)
            # loss = loss_criterion2(student_logits_t, target)
            y = one_hot_embedding(target, num_classes)
            nirho_coeff = 1
            loss = criterion(student_logits_t, y.float(), epoch, num_classes, 10, nirho_coeff, device)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            _, preds = torch.max(student_logits_t.data, 1)
            match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
            acc = torch.mean(match)
            evidence = relu_evidence(student_logits_t)

            all_evidence = torch.cat((all_evidence, evidence), dim=0)
            total_evidence = torch.sum(evidence, 1, keepdim=True)
            mean_evidence = torch.mean(total_evidence)
            mean_evidence_succ = torch.sum(torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
            mean_evidence_fail = torch.sum(torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (
                        torch.sum(torch.abs(1 - match)) + 1e-20)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().data.numpy())
            all_patches.extend(patch_id)
            all_slides.extend(slide_id)

            probs = torch.nn.functional.softmax(student_logits_t.data, dim=1).cpu().numpy()
            all_outputs0.extend(probs[:, 0])
            all_outputs1.extend(probs[:, 1])
            all_outputs2.extend(probs[:, 2])
            all_outputs3.extend(probs[:, 3])
            all_outputs4.extend(probs[:, 4])
            all_outputs5.extend(probs[:, 5])
            all_outputs6.extend(probs[:, 6])
            all_outputs7.extend(probs[:, 7])
            all_outputs8.extend(probs[:, 8])
            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(student_logits_t, dim=-1, descending=True)
            total_correct += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            F1 = f1_score(all_labels, all_preds, average='weighted')
            data_bar.set_description(
                f'{"Train" if is_train else "Test"} Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f} ACC: {total_correct / total_num * 100:.2f}% f1: {F1 * 100:.2f}%')

    alpha = all_evidence + 1
    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    u = u.cpu().data.numpy()
    prob = prob.cpu().data.numpy()
    u2 = u[:, 0]
    prob_0 = prob[:, 0]
    prob_1 = prob[:, 1]
    prob_2 = prob[:, 2]
    prob_3 = prob[:, 3]
    prob_4 = prob[:, 4]
    prob_5 = prob[:, 5]
    prob_6 = prob[:, 6]
    prob_7 = prob[:, 7]
    prob_8 = prob[:, 8]

    df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_preds,
        'slide_id': all_slides,
        'patch_id': all_patches,
        'probabilities_0': all_outputs0,
        'probabilities_1': all_outputs1,
        'probabilities_2': all_outputs2,
        'probabilities_3': all_outputs3,
        'probabilities_4': all_outputs4,
        'probabilities_5': all_outputs5,
        'probabilities_6': all_outputs6,
        'probabilities_7': all_outputs7,
        'probabilities_8': all_outputs8,
        'uncertainty': u2,
        'prob_0': prob_0,
        'prob_1': prob_1,
        'prob_2': prob_2,
        'prob_3': prob_3,
        'prob_4': prob_4,
        'prob_5': prob_5,
        'prob_6': prob_6,
        'prob_7': prob_7,
        'prob_8': prob_8
    })

    return total_loss / total_num, total_correct / total_num * 100, df, F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    group_modelset = parser.add_mutually_exclusive_group(required=True)
    group_modelset.add_argument('--model_path', type=str,
                                default='/n/holyscratch01/wadduwage_lab/uom_hasindri/simCLRv2_nct100k/nct1k_finetune_log/linear_model.pth',
                                help='The pretrained model path')
    group_modelset.add_argument('--pre_model_path', type=str,
                                default='/n/holyscratch01/wadduwage_lab/uom_hasindri/simCLRv2_nct100k/logs2/128_0.5_256_500_model_500.pth',
                                help='The pretrained model path')
    group_modelset.add_argument("--random", action="store_true", default=False,
                                help="No pre-training, use random weights")
    group_modelset.add_argument("--pretrained", action="store_true", default=False,
                                help="Use Imagenet pretrained Resnet")

    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    parser.add_argument('--training_data_csv',
                        default='/n/holyscratch01/wadduwage_lab/uom_hasindri/simCLRv2_nct100k/nct100k.csv',
                        required=True, type=str,
                        help='Path to file to use to read training data')
    parser.add_argument('--test_data_csv',
                        default='/n/holyscratch01/wadduwage_lab/uom_hasindri/simCLRv2_nct100k/crc7k.csv', required=True,
                        type=str, help='Path to file to use to read test data')
    # For validation set, need to specify either csv or train/val split ratio
    group_validationset = parser.add_mutually_exclusive_group(required=True)
    group_validationset.add_argument('--validation_data_csv',
                                     default='/n/holyscratch01/wadduwage_lab/uom_hasindri/simCLRv2_nct100k/crc7k.csv',
                                     type=str,
                                     help='Path to file to use to read validation data')
    group_validationset.add_argument('--trainingset_split', type=float,
                                     help='If not none, training csv with be split in train/val. Value between 0-1')
    parser.add_argument("--num_classes", type=int, default=9, help="Number of classes")
    parser.add_argument("--uncertainty", type=bool, default=False, help="Uncertainty")
    parser.add_argument('--dataset', choices=['cam', 'patchcam', 'cam_rgb_hed', 'ovary', 'skin', 'nct100k', 'mouse'],
                        default='nct100k',
                        type=str, help='Dataset')
    parser.add_argument('--data_input_dir', default='/n/holyscratch01/wadduwage_lab/Nirho', type=str,
                        help='Base folder for images')
    parser.add_argument('--data_input_dir_test', type=str, required=False, help='Base folder for images')
    parser.add_argument('--save_dir',
                        default='/n/holyscratch01/wadduwage_lab/uom_hasindri/simCLRv2_nct100k/distillation_log',
                        type=str, help='Path to save log')
    parser.add_argument('--save_after', type=int, default=1,
                        help='Save model after every Nth epoch, default every epoch')
    parser.add_argument("--balanced_validation_set", action="store_true", default=False,
                        help="Equal size of classes in validation AND test set", )

    parser.add_argument("--finetune", action="store_true", default=False,
                        help="If true, pre-trained model weights will not be frozen.")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (l2 reg)')
    parser.add_argument("--model_to_save", choices=['best', 'latest'], default='best', type=str,
                        help='Save latest or best (based on val acc)')
    parser.add_argument('--seed', type=int, default=44, help='seed')

    parser.add_argument("--use_album", action="store_true", default=False,
                        help="use Albumentations as augmentation lib", )
    parser.add_argument("--balanced_training_set", action="store_true", default=False,
                        help="Equal size of classes in train - SUPERVISED!")

    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help="Choice of optimizer")

    # Common augmentations
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--scale", nargs=2, type=float, default=[0.2, 1.0])
    # RGB augmentations
    parser.add_argument("--rgb_gaussian_blur_p", type=float, default=0,
                        help="probability of using gaussian blur (only on rgb)")
    parser.add_argument("--rgb_jitter_d", type=float, default=1, help="color jitter 0.8*d, val 0.2*d (only on rgb)")
    parser.add_argument("--rgb_jitter_p", type=float, default=0.8,
                        help="probability of using color jitter(only on rgb)")
    parser.add_argument("--rgb_contrast", type=float, default=0.2, help="value of contrast (rgb only)")
    parser.add_argument("--rgb_contrast_p", type=float, default=0, help="prob of using contrast (rgb only)")
    parser.add_argument("--rgb_grid_distort_p", type=float, default=0,
                        help="probability of using grid distort (only on rgb)")
    parser.add_argument("--rgb_grid_shuffle_p", type=float, default=0,
                        help="probability of using grid shuffle (only on rgb)")

    opt = validate_arguments(parser.parse_args())
    temperature = 1.

    # opt.output_dims = 2048
    opt.output_dims = 1024

    is_windows = True if os.name == 'nt' else False
    opt.num_workers = 0 if is_windows else 40

    opt.train_supervised = True
    opt.grayscale = False

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    opt.log_path = opt.save_dir

    # Write the parameters used to run experiment to file
    with open(f'{opt.log_path}/metadata_train.txt', 'w') as metadata_file:
        metadata_file.write(json.dumps(vars(opt)))

    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', opt.device)

    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_path, batch_size, epochs = opt.model_path, opt.batch_size, opt.epochs

    model_t = Net(opt)

    model_s = Net(opt)

    model_t, num_GPU = distribute_over_GPUs(opt, model_t)
    model_t.load_state_dict(torch.load(f'{opt.model_path}'))
    model_t.eval()

    model_s, num_GPU = distribute_over_GPUs(opt, model_s)
    model_s.load_state_dict(torch.load(f'{opt.model_path}'))

    train_loader, train_data, val_loader, val_data, test_loader, test_data = get_dataloader(opt)

    if not opt.finetune:
        for param in model_t.module.f.parameters():
            param.requires_grad = False

    if opt.optimizer == 'adam':
        optimizer = optim.Adam(model_s.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        optimizer = optim.SGD(model_s.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
                              momentum=0.9, nesterov=True)

    scheduler = CosineAnnealingLR(optimizer, opt.epochs)
    uncertain = opt.uncertainty
    if uncertain:
        criterion = edl_mse_loss
    else:
        loss_criterion = JointsMSELoss(use_target_weight=False).to(opt.device)

    loss_criterion2 = nn.CrossEntropyLoss()

    loss_criterion3 = nn.MSELoss(size_average=True).to(opt.device)
    results = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'F1_score': []}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, F1 = train_val(model_t, model_s, train_loader, optimizer, opt.device)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        val_loss, val_acc, _, F1 = test_val(model_s, val_loader, None, opt.device)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['F1_score'].append(F1)

        scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'{opt.log_path}/linear_statistics.csv', index_label='epoch')

        if opt.model_to_save == 'best' and val_acc > best_acc:
            # Save only the if the accuracy exceeds previous accuracy
            best_acc = val_acc
            torch.save(model_s.state_dict(), f'{opt.log_path}/linear_model2.pth')
        elif opt.model_to_save == 'latest':
            # Save latest model
            best_acc = val_acc
            torch.save(model_s.state_dict(), f'{opt.log_path}/linear_model2.pth')

    # trainig finished, run test
    print('Training finished, testing started...')
    # Load saved model
    model_s.load_state_dict(torch.load(f'{opt.log_path}/linear_model2.pth'))
    model_s.eval()
    test_loss, test_acc, df, F1 = test_val(model_s, test_loader, None, opt.device)
    test_stat = [test_loss, test_acc, F1]
    test_statis = pd.DataFrame(data=test_stat)
    test_statis.to_csv(f"{opt.log_path}/test_stat.csv", index=False, header=None)

    df.to_csv(
        f"{opt.log_path}/inference_result_model.csv")