import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .cross_attention import CrossAttentionModule

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, return_all_tokens=False): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        #code cũ
        # device = x.device
        # batch_indices = torch.arange(x.shape[0], device=device)
        # index = tokenized_prompts.argmax(dim=-1).to(device)

        # x = x[batch_indices, index] @ self.text_projection
        # return x
        if return_all_tokens:
            # Trả về tất cả tokens (N, L, D)
            x = x @ self.text_projection  # Project tất cả tokens
            return x
        else:
            # Chỉ trả về [EOS] token (tương thích với code cũ)
            device = x.device
            batch_indices = torch.arange(x.shape[0], device=device)
            index = tokenized_prompts.argmax(dim=-1).to(device)
            return x[batch_indices, index] @ self.text_projection
#--------------------------------------------------------------------------------------------------
#
#
# ===================== THÊM CÁC CLASS MỚI =====================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory):
        # tgt: (B, L_t, D), memory: (B, L_v, D)
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # Cross-attention (tgt query, memory key/value)
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

class TextGuidedDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, text_tokens, visual_tokens):
        out = text_tokens
        for layer in self.layers:
            out = layer(out, visual_tokens)
        return self.norm(out)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        self.use_cross_attention = cfg.MODEL.USE_CROSS_ATTENTION

        # ---------- classifier ----------
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # ---------- bottleneck ----------
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # ---------- CLIP backbone ----------
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(
            self.model_name,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size
        )
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        # ---------- SIE embedding ----------
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('view number is : {}'.format(view_num))

       # ---------- prompt & text encoder ----------
        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(
            num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding
        )
        self.text_encoder = TextEncoder(clip_model)

        # ---------- simplified prompt ----------
        # Thay vì chỉ lưu embedding, encode luôn text tokens
        simple_prompt = "A photo of a person"
        simple_tokenized = clip.tokenize(simple_prompt).cuda()
        with torch.no_grad():
            simple_embedding = clip_model.token_embedding(simple_tokenized).type(clip_model.dtype)
            simple_text_tokens = self.text_encoder(simple_embedding, simple_tokenized, return_all_tokens=True).cpu()

        self.register_buffer("simple_text_tokens", simple_text_tokens)

        # ---------- cross-attention module ----------
        if self.use_cross_attention:
            self.text_guided_decoder = TextGuidedDecoder(
                num_layers=cfg.MODEL.CROSS_ATTENTION_LAYERS,  # nên = 3
                d_model=cfg.MODEL.CROSS_ATTENTION_DIM,        # 512
                nhead=cfg.MODEL.CROSS_ATTENTION_HEADS,        # 8
                dim_feedforward=cfg.MODEL.CROSS_ATTENTION_FFN_DIM,  # 2048
                dropout=0.1
            )
            self.ca_alpha = nn.Parameter(torch.tensor(0.1))   # λ trong paper
            
    def forward(self, x=None, label=None, get_image=False, get_text=False,  
        cam_label=None, view_label=None, use_cross_attention=False, return_all=False):
        # ====== TEXT FORWARD (Stage1/Stage2 precompute prototype) ======
        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts, return_all_tokens=False)
            return text_features
            
        # ====== IMAGE-ONLY FEATURES (Stage1) ======
        if get_image:
            # Stage1 không dùng SIE, không dùng CA
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        # ====== MAIN FORWARD ======
        if self.model_name == 'RN50':
            # ---------- RN50 branch: không dùng CA ----------
            image_features_last, image_features, image_features_proj = self.image_encoder(x)

            img_feature_last = nn.functional.avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(x.shape[0], -1)

            img_feature_proj = image_features_proj[0]   # (B, 512)

            # pre/post-CA giống nhau (không có CA cho RN50)
            feat_pre_ca = img_feature_proj
            feat_post_ca = img_feature_proj

        elif self.model_name == 'ViT-B-16':
            # ---------- SIE embedding ----------
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            # ---------- visual encoder ----------
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)

            # CLS global features
            img_feature_last = image_features_last[:, 0]      # (B, 768)
            img_feature = image_features[:, 0]                # (B, 768)

            # Lấy patch tokens (bỏ CLS token)
            patch_tokens = image_features_proj[:, 1:, :]  # (B, num_patches, 512)
            cls_token = image_features_proj[:, 0, :]      # (B, 512)

            # Text tokens
            if self.training and (label is not None):
                prompts = self.prompt_learner(label)
                # Lấy tất cả text tokens (N, L, 512)
                text_tokens = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts, return_all_tokens=True)
            else:
                # Eval/Inference: dùng prompt cố định (đã lưu sẵn)
                text_tokens = self.simple_text_tokens.to(patch_tokens.device)  # (1, L, 512)
                text_tokens = text_tokens.expand(patch_tokens.size(0), -1, -1)  # (B, L, 512)

            # Feature trước CA (dùng cho I2T loss)
            feat_pre_ca = cls_token

            # Cross-Attention: text query, visual key/value
            if self.use_cross_attention and use_cross_attention:
                refined_text_tokens = self.text_guided_decoder(text_tokens, patch_tokens)  # (B, L, 512)
                # Lấy EOS token (vị trí cuối cùng) - vì CLIP thêm EOS ở cuối
                refined_eos = refined_text_tokens[:, -1, :]   # (B, 512)
                # Text feature gốc (EOS)
                orig_eos = text_tokens[:, -1, :]               # (B, 512)
                # Residual connection (công thức 12)
                joint_feat = orig_eos + self.ca_alpha * refined_eos
                feat_post_ca = joint_feat
            else:
                feat_post_ca = feat_pre_ca

        # ====== NECK (bottleneck) ======
        # 768-d visual cho Triplet
        feat = self.bottleneck(img_feature)           # (B, in_planes)

        # 512-d projection sau interaction cho ID + metric
        feat_proj = self.bottleneck_proj(feat_post_ca)  # (B, 512)

        if self.training or return_all:   
            # ID loss dùng feat_proj sau CA
            cls_score = self.classifier(feat)    
            cls_score_proj = self.classifier_proj(feat_proj)
            # Trả về:
            #   - cls_score_proj: cho L_id (sau CA)
            #   - feat_post_ca:   feature sau CA cho L_tri
            #   - feat_pre_ca:    feature trước CA cho Li2t_ce / SupCon
            return [cls_score, cls_score_proj],[img_feature_last, img_feature, feat_post_ca], feat_pre_ca
        else:
            # ===== TEST / EVAL =====
            if self.neck_feat == 'after':
                # dùng feature sau BN (cả 768 + 512)
                return torch.cat([feat, feat_proj], dim=1)
            else:
                # "before": vẫn dùng feat_post_ca (sau interaction) thay vì img_feature_proj thuần
                return torch.cat([img_feature, feat_post_ca], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx
        

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 

