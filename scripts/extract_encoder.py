import argparse
import torch


def extract_feature_encoder(path, modal='image'):
    obj = torch.load(path)
    state_dict = obj['state_dict']

    output_state = {}
    if modal == 'image':
        for key, val in state_dict.items():
            if key.startswith('token_type_embeddings.') or \
                key.startswith('transformer.'):
                
                output_state[key] = val
        
        torch.save(output_state, 'checkpoints/0-feature_extractor/image_encoder.ckpt')
        
    elif modal == 'text':
        for key, val in state_dict.items():
            if key.startswith('token_type_embeddings.') or \
                key.startswith('transformer.') or \
                key.startswith('text_embeddings.'):
               
                output_state[key] = val
        
        torch.save(output_state, 'checkpoints/0-feature_extractor/text_encoder.ckpt')
    else:
        raise ValueError("Unknown modal:{}".format(modal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path of pre-trained ViLT model')
    parser.add_argument('--modal', default='image', type=str, help='specify modality for encoder')

    args = parser.parse_args()

    extract_feature_encoder(args.path, args.modal)