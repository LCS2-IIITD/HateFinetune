from all_imports import *

file_name = __file__.rsplit("/", 1)[1].split('.')[0]
print(file_name)

with open('config.json') as f:
    data = json.load(f)

seeds = [12, 127, 451]
keys = ['hatexplain_label', 'olid_taska', 'waseem', 'davidson', 'toxigen_label', 'founta', 'dynabench_label']
for seed in seeds:
    set_random_seed(seed)
    for dset in keys:
        TASK = data[dset]['TASK']
        dataset_name = data[dset]['dataset_name']
        replacement_dict = data[dset]['replacement_dict']
        SAVE_PATH = f"saves/{TASK}/{seed}/"
        device_used = "cuda:0" if torch.cuda.is_available() else "cpu"

        models = ["bert-base-uncased", "GroNLP/hateBERT", "bert-base-multilingual-cased"]
        model_names = ["bert-base-uncased", "hateBERT", "bert-base-multilingual-cased"]

        if not os.path.exists(f"saves/{TASK}"):
            os.mkdir(f"saves/{TASK}")
        if not os.path.exists(f"saves/{TASK}/{seed}"):
            os.mkdir(f"saves/{TASK}/{seed}")
        for model_name in model_names:
            subfolder = SAVE_PATH + model_name + '/'
            if not os.path.exists(subfolder):
                os.mkdir(subfolder)
            for i in range(12):
                if not os.path.exists(subfolder + f'frozen_except_{i}/'):
                    os.mkdir(subfolder + f'frozen_except_{i}/')

        # Load the data
        try:
            train, validation, test = load_custom_ds(dataset_name)
        except:
            # delete the folder
            continue

        train['label'] = train['label'].replace(replacement_dict)
        validation['label'] = validation['label'].replace(replacement_dict)
        test['label'] = test['label'].replace(replacement_dict)

        train = train[['text', 'label']]
        validation = validation[['text', 'label']]
        test = test[['text', 'label']]

        print(len(train), len(validation), len(test))
        
        train = train.dropna()
        validation = validation.dropna()
        test = test.dropna()
        
        print(len(train), len(validation), len(test))
        
        dataset_hf = DatasetDict({
            'train': Dataset.from_pandas(train),
            'test': Dataset.from_pandas(test),
            'validation': Dataset.from_pandas(validation),
        })
        
        c = -1

        for model_name in models:
            c += 1
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            device = torch.device(device_used)

            for i in range(12):
                CURRENT_SAVE_PATH = SAVE_PATH + model_names[c] + '/' + f'frozen_except_{i}/'
                print(CURRENT_SAVE_PATH)
                # if folder not empty, skip
                if len(os.listdir(CURRENT_SAVE_PATH)) > 0:
                    print("skipping", CURRENT_SAVE_PATH)
                    continue

                dataset_hf_tokenized = dataset_hf.map(lambda examples: 
                                        tokenizer(
                                            examples['text'],
                                            truncation=True, 
                                            padding='max_length'), 
                                        batched=True, batch_size=16)

                dataset_hf_tokenized.set_format("torch",columns=["input_ids", "attention_mask", "label"])

                train_dataloader, val_dataloader, test_dataloader = prepare_data_loaders(tokenizer, dataset_hf_tokenized)
                model = ClassifierOnMiddleLayer(model_name, i, len(replacement_dict))

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

                # pickle dump the model
                with open(CURRENT_SAVE_PATH + f"model_bertweet_{i}_{seed}", "wb") as f:
                    pickle.dump(model, f)
