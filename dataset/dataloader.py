from .base import BaseDataset, PILDataset
import numpy as np
import scipy.io as scio


def split_data(captions, indexs, labels, query_num=5000, train_num=10000, seed=None, unseen=False, unseen_ratio=0.25, query_per_class=100):
    np.random.seed(seed=seed)
    N, C = labels.shape

    if not unseen:
        random_index = np.random.permutation(range(len(indexs)))
        query_index = random_index[: query_num]
        train_index = random_index[query_num: query_num + train_num]
        retrieval_index = random_index[query_num:]

        query_indexs = indexs[query_index]
        query_captions = captions[query_index]
        query_labels = labels[query_index]

        train_indexs = indexs[train_index]
        train_captions = captions[train_index]
        train_labels = labels[train_index]

        retrieval_indexs = indexs[retrieval_index]
        retrieval_captions = captions[retrieval_index]
        retrieval_labels = labels[retrieval_index]

        split_indexs = (query_indexs, train_indexs, retrieval_indexs)
        split_captions = (query_captions, train_captions, retrieval_captions)
        split_labels = (query_labels, train_labels, retrieval_labels)
        return split_indexs, split_captions, split_labels

    else:
        all_classes = np.arange(C)
        np.random.shuffle(all_classes)

        fold_size = int(C * unseen_ratio)
        class_folds = [all_classes[i * fold_size:(i + 1) * fold_size] for i in range(4)]
        remain = set(all_classes) - set(np.concatenate(class_folds))
        if remain:
            class_folds[-1] = np.concatenate([class_folds[-1], list(remain)])

        split_indexs_list, split_captions_list, split_labels_list = [], [], []

        for fold_id, unseen_classes in enumerate(class_folds):
            unseen_classes = set(unseen_classes)
            seen_classes = set(all_classes) - unseen_classes

            seen_mask = np.array([set(np.where(lbl > 0)[0]).issubset(seen_classes) for lbl in labels])
            unseen_mask = np.array([set(np.where(lbl > 0)[0]).issubset(unseen_classes) for lbl in labels])

            seen_index = np.where(seen_mask)[0]
            unseen_index = np.where(unseen_mask)[0]

            # ---- seen 训练样本 ----
            np.random.shuffle(seen_index)
            train_index = seen_index[:train_num]

            # ---- unseen → query / retrieval ----
            query_index, retrieval_index = [], []
            for c in unseen_classes:
                sample_idx = np.where(labels[unseen_index, c] > 0)[0]
                if len(sample_idx) == 0:
                    continue
                np.random.shuffle(sample_idx)
                q_num = min(query_per_class, len(sample_idx) // 2)
                query_index.extend(unseen_index[sample_idx[:q_num]])
                retrieval_index.extend(unseen_index[sample_idx[q_num:]])

            query_index = np.unique(np.array(query_index))
            retrieval_index = np.unique(np.array(retrieval_index))

            print(f"[Fold {fold_id + 1}/4] Seen train: {len(train_index)}, "
                  f"Unseen query: {len(query_index)}, Unseen db: {len(retrieval_index)}")

            # ---- 构建折数据 ----
            query_indexs = indexs[query_index]
            train_indexs = indexs[train_index]
            retrieval_indexs = indexs[retrieval_index]

            query_captions = captions[query_index]
            train_captions = captions[train_index]
            retrieval_captions = captions[retrieval_index]

            query_labels = labels[query_index]
            train_labels = labels[train_index]
            retrieval_labels = labels[retrieval_index]

            split_indexs = (query_indexs, train_indexs, retrieval_indexs)
            split_captions = (query_captions, train_captions, retrieval_captions)
            split_labels = (query_labels, train_labels, retrieval_labels)

            split_indexs_list.append(split_indexs)
            split_captions_list.append(split_captions)
            split_labels_list.append(split_labels)

        return split_indexs_list, split_captions_list, split_labels_list


def dataloader(captionFile: str,
                indexFile: str,
                labelFile: str,
                maxWords=32,
                imageResolution=224,
                query_num=5000, 
                train_num=10000, 
                seed=None,
                clip=True,
                wiki=False,
                unseen=False,
                npy=False):
    if captionFile.endswith("mat"):
        captions = scio.loadmat(captionFile)["caption"]
        captions = captions[0] if captions.shape[0] == 1 else captions
    elif captionFile.endswith("txt"):
        with open(captionFile, "r") as f:
            captions = f.readlines()
        captions = np.asarray([[item.strip()] for item in captions])
    else:
        raise ValueError("the format of 'captionFile' doesn't support, only support [txt, mat] format.")
    if not npy:
        indexs = scio.loadmat(indexFile)["index"]
    else:
        indexs = np.load(indexFile, allow_pickle=True)
    labels = scio.loadmat(labelFile)["category"]

    split_indexs, split_captions, split_labels = split_data(captions, indexs, labels, query_num=query_num, train_num=train_num, seed=seed, unseen=unseen)

    # ==============================
    # 普通模式：返回单个数据集
    # ==============================
    # if not unseen:
    Dataset = BaseDataset if clip else PILDataset
    collate_fn = clip_collate_fn if clip else PIL_collate_fn

    if wiki:
        train_idx = np.load("wiki_train_idx.npy")
        test_idx = np.load("wiki_test_idx.npy")

        train_data = Dataset(
            captions=captions[train_idx],
            indexs=indexs[train_idx],
            labels=labels[train_idx],
            maxWords=maxWords,
            imageResolution=imageResolution,
            npy=npy
        )

        # Wiki protocol: test set is both query & retrieval
        query_data = Dataset(
            captions=captions[test_idx],
            indexs=indexs[test_idx],
            labels=labels[test_idx],
            maxWords=maxWords,
            imageResolution=imageResolution,
            is_train=False,
            npy=npy
        )

        retrieval_data = Dataset(
            captions=captions[test_idx],
            indexs=indexs[test_idx],
            labels=labels[test_idx],
            maxWords=maxWords,
            imageResolution=imageResolution,
            is_train=False,
            npy=npy
        )

        return train_data, query_data, retrieval_data, collate_fn


    train_data = Dataset(captions=split_captions[1], indexs=split_indexs[1], labels=split_labels[1],
                             maxWords=maxWords, imageResolution=imageResolution, npy=npy)
    query_data = Dataset(captions=split_captions[0], indexs=split_indexs[0], labels=split_labels[0],
                             maxWords=maxWords, imageResolution=imageResolution, is_train=False, npy=npy)
    retrieval_data = Dataset(captions=split_captions[2], indexs=split_indexs[2], labels=split_labels[2],
                                 maxWords=maxWords, imageResolution=imageResolution, is_train=False, npy=npy)

    return train_data, query_data, retrieval_data, collate_fn

# --------------------------------------------------------------------------------------------------------------------------
    # ==============================
    # Unseen 模式：返回四折列表
    # ==============================
    # train_data_list, query_data_list, retrieval_data_list = [], [], []
    #
    # for fold_id in range(4):
    #     split_index = split_indexs[fold_id]
    #     split_caption = split_captions[fold_id]
    #     split_label = split_labels[fold_id]
    #
    #     train_data = BaseDataset(captions=split_caption[1], indexs=split_index[1], labels=split_label[1],
    #                              maxWords=maxWords, imageResolution=imageResolution, npy=npy)
    #     query_data = BaseDataset(captions=split_caption[0], indexs=split_index[0], labels=split_label[0],
    #                              maxWords=maxWords, imageResolution=imageResolution, is_train=False, npy=npy)
    #     retrieval_data = BaseDataset(captions=split_caption[2], indexs=split_index[2], labels=split_label[2],
    #                                  maxWords=maxWords, imageResolution=imageResolution, is_train=False, npy=npy)
    #
    #     train_data_list.append(train_data)
    #     query_data_list.append(query_data)
    #     retrieval_data_list.append(retrieval_data)
    #
    # return train_data_list, query_data_list, retrieval_data_list

import torch

def PIL_collate_fn(batch):
    """
    处理 SigLIP 数据集的 collate 函数
    批量中的每个元素是 (PIL图像, 文本, 标签, 索引)
    """
    images, texts, labels, indices = zip(*batch)

    # 标签和索引可以直接堆叠
    labels = torch.stack(labels)
    indices = torch.stack(indices)

    # 返回 PIL 图像列表和文本列表，让 SigLIP 的 processor 在模型中处理
    return list(images), list(texts), labels, indices


def clip_collate_fn(batch):
    """
    处理 CLIP 数据集的默认 collate 函数
    """
    return torch.utils.data.dataloader.default_collate(batch)
