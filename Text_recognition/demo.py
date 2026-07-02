
import torch
from torch.autograd import Variable
from time import time
import Text_recognition.models.crnn as crnn
# from Text_recognition

def load_model(model_path: str = './data/crnn.pth'):
    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    # print(f'Loading pretrained model from {model_path}')
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    return model


def extract_text_from_image(image, model, converter, transformer) -> dict:
    # time_start = time()


    answer_text = ""

    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    
    image = image.view(1, *image.size())
    image = Variable(image)

    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print('time: %.2f s' % (time() - time_start))
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    answer_text += f"{sim_pred} "
    
    return answer_text

