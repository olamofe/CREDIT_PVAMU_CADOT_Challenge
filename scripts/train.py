
import sys
sys.path.insert(0, 'CREDIT_PVAMU_CADOT_Challenge/ultralytics')
from comet_ml import Experiment, init
#from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO
#from sahi import AutoDetectionModel
#from sahi.utils.cv import read_image
#from sahi.utils.file import download_from_url
#from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
#from IPython.display import Image
import yaml
import torch
import os
import select_available_gpu
import argparse


# Enable mixed precision training
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['TORCH_AUTOCast'] = '1'


project_root = Path(__file__).resolve().parents[1]

#print(f'======== project_root = {project_root} ======')

# Load training config
with open(f'{project_root}/models/train_config.yaml', 'r') as f:
    train_cfg = yaml.safe_load(f)

datapath = f'{project_root}/data/config.yaml'


def model_info(path='', model_type='YOLO', model_variant='', scale='x',
                datasets='CADOT_Challenge', data_size='500',
                freeze=5, pretrained=True, pretrained_data_name='xview'
                ):
         
    tasks_name = f'Model_{model_type}_Variant_{model_variant}_Scale_{scale}_Input_DataSize_{data_size}_Freeze_{freeze}_layers'
    experiment_tasks =  f'{tasks_name}_Pretrained_{pretrained_data_name}_Dataset' if pretrained else tasks_name
    global_project = f'{model_type}_Models_Trained_On_{data_size}_Input_{datasets}_Data_With_Augumentation'
    
    
    model_name = f'Model_{model_type}_Variant_{model_variant}_Scale_{scale}'

    return experiment_tasks, global_project, model_name
    
    




def create_comet_instance_task(task_name='', global_project=''):
    experiment_ = Experiment(
                    api_key="*********",
                    project_name=global_project,
                    workspace="myworks_space"
                    )
    # Set the experiment name
    experiment_.set_name(task_name)

    init()
    return experiment_

def train_model(args, experiment_tasks='', global_project='', 
                model_name=''):

    
    #experiment = create_comet_instance_task(task_name=experiment_tasks, global_project=global_project)

    # Use the selected GPU
    selected_device = select_available_gpu.get_best_device()
    print(f'Selected device is GPU :: {selected_device} |  type = {type(selected_device)}')

    device = torch.device(selected_device)
    torch.cuda.set_device(device)
    pretained_path = Path(os.path.join(project_root, args.pretrained_path))
    print(f'model pretrained weigh pth = {pretained_path}')
    model = YOLO(pretained_path)

    model.to(device)

    yaml_path = Path(os.path.join(project_root, args.datapath))
    print(f'yaml_path pth = {yaml_path}')

    train_project_dir = Path(os.path.join(project_root, 'models/CADOT_Trained_Models'))
    
    '''
    results = model.train(data=yaml_path, 
                        epochs=args.epochs, 
                        project=train_project_dir,
                        name=model_name, 
                        batch=args.batch_size,
                        imgsz=args.imgsz,
                        dropout=args.dropout,
                        freeze=args.freeze,
                        save_period=save_period
                        )
    '''

    # Train the model
    results = model.train(
                            data=yaml_path,                   # your dataset config file
                            imgsz=train_cfg['imgsz'],
                            epochs=train_cfg['epochs'],
                            batch=train_cfg['batch'],
                            optimizer=train_cfg['optimizer'],
                            freeze=args.freeze,
                            copy_paste=train_cfg['copy_paste'],
                            lr0=train_cfg['lr0'],
                            lrf=train_cfg['lrf'],
                            momentum=train_cfg['momentum'],
                            weight_decay=train_cfg['weight_decay'],
                            warmup_epochs=train_cfg['warmup_epochs'],
                            warmup_momentum=train_cfg['warmup_momentum'],
                            warmup_bias_lr=train_cfg['warmup_bias_lr'],
                            box=train_cfg['box'],
                            cls=train_cfg['cls'],
                            dfl=train_cfg['dfl'],
                            degrees=train_cfg['degrees'],
                            translate=train_cfg['translate'],
                            scale=train_cfg['scale'],
                            shear=train_cfg['shear'],
                            perspective=train_cfg['perspective'],
                            flipud=train_cfg['flipud'],
                            fliplr=train_cfg['fliplr'],
                            mosaic=train_cfg['mosaic'],
                            mixup=train_cfg['mixup'],
                            name=model_name,
                            project=train_project_dir,
                            save_period=train_cfg['save_period'],
                            save=train_cfg['save'],
                            val=train_cfg['val']
                     )      

def main(args):
    #print('CALLED')
    #model_variant = 'yolofocus11' if 'focus' in args.pretrained_path else 'yolo11
    
    version = '8' if '8' in args.pretrained_path else ('11' if '11' in args.pretrained_path else 12)
    
    model_variant = 'yolo'
    
    
    if 'focus' in args.pretrained_path:
        model_variant = 'yolofocus'
    
    if 'FCG' in args.pretrained_path:
        model_variant = 'yoloFCG'
    
    model_variant = f'{model_variant}{version}'

    scale = 'x' if 'x' in args.pretrained_path else 'l'

    experiment_tasks, global_project, model_name = model_info(path=args.pretrained_path, freeze=args.freeze,
                                                            model_variant=model_variant, scale=scale)
    
    train_model(args, experiment_tasks=experiment_tasks, 
                    global_project=global_project, model_name=model_name)

'''
def run_models():
    for model_path in models_path_list:
        model_variant = 'yolofocus11' if 'focus' in model_path else 'yolo11'
        scale = 'x' if 'x' in model_path else l

        task_name, experiment_tasks, global_project, model_name = model_info(path=model_path, 
                                                                            model_variant=model_variant, scale=scale)
        
        train_model(model_path=model_path, experiment_tasks=experiment_tasks, 
                    global_project=global_project, model_name=model_name)
'''



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="YOLO training script")

    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to pretrained model weights")
    parser.add_argument("--datapath", type=str, default="data/data.yaml", required=True, help="Path to YAML dataset file (e.g., data.yaml)")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to freeze")


    args = parser.parse_args()
    main(args)

