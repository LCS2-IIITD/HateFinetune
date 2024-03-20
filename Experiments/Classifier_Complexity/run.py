from all_imports import *
import numpy as np
file_name = __file__.rsplit("/", 1)[1].split('.')[0]

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

        print(data[dset])
        SAVE_PATH = f"saves/{TASK}/{seed}/"
        device_used = "cuda:0" if torch.cuda.is_available() else "cpu"

        models = ['bert-base-uncased', 'vinai/bertweet-base', 'GroNLP/hateBERT', 'bert-base-multilingual-cased']
        model_names = ['bert-base-uncased', 'bertweet-base', 'hateBERT', 'bert-base-multilingual-cased']

        classifiers = ['simple', 'medium', 'complex']
        classifier_names = ['simple', 'medium', 'complex']

        print(models)

        # if os.path.exists(SAVE_PATH):
        #     pass
        # else:
        if not os.path.exists(f"saves/{TASK}"):
            os.mkdir(f"saves/{TASK}")
        if not os.path.exists(f"saves/{TASK}/{seed}"):
            os.mkdir(f"saves/{TASK}/{seed}")
        for model_name in model_names:
            subfolder = SAVE_PATH + model_name + '/'
            print(subfolder)
            if not os.path.exists(subfolder):
                print("Creating folder ", subfolder)
                os.mkdir(subfolder)
            for classifier_name in classifier_names:
                subsubfolder = subfolder + classifier_name + '/'
                if not os.path.exists(subsubfolder):
                    print("Creating folder ", subsubfolder)
                    os.mkdir(subsubfolder)
        # Load the data
        train, validation, test = load_custom_ds(dataset_name)

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
            c2 = -1
            for classifier in classifiers:
                c2 += 1
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                CURRENT_SAVE_PATH = SAVE_PATH + model_names[c] + '/' + classifier + '/'
                # if current save path not empty, then skip
                # if len(list(os.listdir(CURRENT_SAVE_PATH))) > 0:
                #     print("Skipping this model ", CURRENT_SAVE_PATH)
                #     continue

                dataset_hf_tokenized = dataset_hf.map(lambda examples: 
                                        tokenizer(
                                            examples['text'],
                                            truncation=True, 
                                            padding='max_length'), 
                                        batched=True, batch_size=16)

                dataset_hf_tokenized.set_format("torch",columns=["input_ids", "attention_mask", "label"])

                device = torch.device(device_used)

                train_dataloader, val_dataloader, test_dataloader = prepare_data_loaders(tokenizer, dataset_hf_tokenized)

                CURRENT_SAVE_PATH = SAVE_PATH + model_names[c] + '/' + classifier + '/'
                print()
                print(CURRENT_SAVE_PATH)
                print(classifier)
                print(model_name)
                print()

                # if current save path not empty, then skip
                # if len(list(os.listdir(CURRENT_SAVE_PATH))) > 0:
                #     print("Skipping this model")
                #     continue

                if classifier == 'simple':
                    model = SimpleModel(model_name, len(replacement_dict))
                elif classifier == 'medium':
                    model = MediumModel(model_name, len(replacement_dict))
                elif classifier == 'complex':
                    model = ComplexModel(model_name, len(replacement_dict))

                model.to(device)

                # Optimizer

                optimizer = AdamW(model.parameters())

                num_epoch = 5

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
        #             print(train_dataloader)
        #             # Recovers the original `dataset` from the `dataloader`
        #             dataset = train_dataloader.dataset
        #             n_samples = len(dataset)

        #             # Get a random sample
        #             random_index = int(np.random.random()*n_samples)
        #             single_example = dataset[random_index]
        #             print(n_samples, single_example)
                    for step, batch in enumerate(train_dataloader):
                        # print(step, batch)
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
                    # Save the metrics to a file
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
