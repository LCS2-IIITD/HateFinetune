from all_imports import *

BASE_PATH = "hateData/"
# list all directories in the base path
import os
dirs = os.listdir(BASE_PATH)
print(dirs)
# keep only 
for direc in dirs:
    SAVE_PATH = f"saves/{direc}/bert_base_uncased"
    
    # if save path contains files, skip
    if os.path.exists(SAVE_PATH):
        if len(os.listdir(SAVE_PATH)) > 0:
            print("Skipping: ", direc)
            continue

    print(direc)
    dir1, dir2 = direc.split('_AND_')

    ls_1 = os.listdir(BASE_PATH + direc + '/' + dir1)
    ls_2 = os.listdir(BASE_PATH + direc + '/' + dir2)

    data_1 = []
    data_2 = []

    for dset in ls_1:
        # open txt file
        with open(BASE_PATH + direc + '/' + dir1 + '/' + dset, 'r') as f:
            data_1.append(f.readlines()[0])
    
    for dset in ls_2:
        # open txt file
        with open(BASE_PATH + direc + '/' + dir2 + '/' + dset, 'r') as f:
            data_2.append(f.readlines()[0])
    
    net_data = data_1 + data_2
    net_labels = [0] * len(data_1) + [1] * len(data_2)

    # create a dataframe
    df = pd.DataFrame({'text': net_data, 'label': net_labels})
    
    SAVE_PATH = f"saves/{direc}"

    if os.path.exists(SAVE_PATH):
        pass
    else:
        os.mkdir(SAVE_PATH)

    

    device_used = "cuda:1" if torch.cuda.is_available() else "cpu"

    models = ["bert-base-uncased"]
    model_names = ["bert_base_uncased"]
                    
    SEED = 42
    set_random_seed(SEED)

    # Load the data
    train, test = train_test_split(df, test_size=0.2, random_state=SEED)
    train, validation = train_test_split(train, test_size=0.2, random_state=SEED)

    dataset_hf = DatasetDict({
        'train': Dataset.from_pandas(train),
        'test': Dataset.from_pandas(test),
        'validation': Dataset.from_pandas(validation),
    })
    
    c = -1

    for model_name in models:
        c += 1
        tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)

        dataset_hf_tokenized = dataset_hf.map(lambda examples: 
                                tokenizer(
                                    examples['text'],
                                    truncation=True, 
                                    padding='max_length'), 
                                batched=True, batch_size=16)

        dataset_hf_tokenized.set_format("torch",columns=["input_ids", "attention_mask", "label"])

        device = torch.device(device_used)

        train_dataloader, val_dataloader, test_dataloader = prepare_data_loaders(tokenizer, dataset_hf_tokenized)

        CURRENT_SAVE_PATH = SAVE_PATH + '/' + model_names[c] + '/'

        if os.path.exists(CURRENT_SAVE_PATH):
            pass
        else:
            os.mkdir(CURRENT_SAVE_PATH)

        model = SimpleModel(model_name, 2)

        model.to(device)

        # Optimizer

        optimizer = AdamW(model.parameters())

        num_epoch = 2

        num_training_steps = num_epoch * len(train_dataloader)

        progress_bar_train = tqdm(range(num_training_steps))
        progress_bar_eval = tqdm(range(num_epoch * len(val_dataloader)))

        lr_scheduler = get_scheduler(
            'linear',
            optimizer = optimizer,
            num_warmup_steps = 10000,
            num_training_steps = num_training_steps,   
        )

        # Training and Saving the model after k steps

        k = 100

        metric_values = []

        for epoch in range(num_epoch):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar_train.update(1)

            # torch.save(model.state_dict(), CURRENT_SAVE_PATH + f"epoch_{epoch}.pt")

            # Calculate the metrics on the validation set
            # model.eval()
            # metric_val = perform_eval_on_validation(val_dataloader, device, model, epoch, progress_bar_eval)        
            # # Save the metrics to a file
            # with open(CURRENT_SAVE_PATH + f"epoch_{epoch}_metric_values_validation.pkl", "wb") as f:
            #     # Save to Pickle
            #     pickle.dump(metric_values, f)

        # Evaluate on the test set
        model.eval()
        test_progress_bar = tqdm(range(len(test_dataloader)))
        test_results = perform_eval_on_test(test_dataloader, device, model, epoch, test_progress_bar)
        # Save the metrics to a file
        with open(CURRENT_SAVE_PATH + f"test_metrics.pkl", "wb") as f:
            # Dump as pickle
            pickle.dump(test_results, f)
