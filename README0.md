

## Record
### Think
* 在空间推理数据集上测试
    * 下载数据集
    * 位置编码的作用
        * 每一层都加位置编码
    * 打乱图像 tokens
    * 打乱文本 tokens

* 多模态幻觉
    * 现象 
        * 无中生有，或装瞎
        * 嵌套关系：图片左边是什么 v.s. 图片中桌子的左边是什么
    * 原因
        * 架构
            * 位置编码
        * 数据
        * 训练
            * 图文没对齐好

### Do
* 4.15
    * 下载数据集：GQA, VQA-v2, POPE; What's UP (-B)
        * √  GQA: https://cs.stanford.edu/people/dorarad/gqa/index.html
        * VQA-v2: 
        * CLEVR: https://cs.stanford.edu/people/jcjohns/clevr/
    * 实验
        * 逐个去除位置编码，观察影响
            * 位置编码包括两部分：(1) ViT; (2) LLMn
                * Qwen2-VL: 2D-RoPR + MRoPE
        * 打乱图像 token 顺序，观察影响
            * 打乱的话，模型会不会在内部重新排列？
        * 打乱图像 token，并逐个去除位置编码，观察影响
* 4.16
    * 实验
        * 删位置编码
            * 逐层删
            * √ 删某层及之后的
                * √ 在 ViT 内删
                * √ 在 LLM 内删
                * √ 在 ViT 和 LLM 内删
        * 打乱图像 tokens 顺序
            * 所有层打乱
                * 打乱之后，位置编码和图像真实位置就不匹配了
                * 打乱之后，性能反增
                    * 关键信息靠近文本的话，性能会不会好一些？
                    * 尝试根据锚框删掉关键区域以外的部分：去噪
                * !! shuffle image tokens，即使给 image tokens 加上了错误的位置编码，对最终性能没有负面影响，这可能是因为在 ViT 阶段已经完成了图像位置信息的融入，后续只是巩固。
                    * !! 为什么加错了位置编码也不影响性能？
                    * 删去 llm pos embed，前几层较为重要，删掉的话会生成 addCriterion。可能是因为影响了语言的建模，即语言生成需要位置编码来保障
                    * 在进 LLM 之前，图像已经由 ViT 编码了，所以在 LLM 层的位置编码可能会冗余
                        * 现在LLM内删的位置编码是同时删文本和图像的，后面还可以做只删图像的，两种设置：
                            * ~~文本从 img_last_token_id + 1 开始——~~ (hijack qwen 里写错了一个，get_rope_index_delete_image_rope 这个函数现在没用了)
                            * 文本保留原有的 position ids (利用 image_mask)
                            * 文本从 0 开始
            * 逐层打乱？
        * 思考
            * 语言的位置编码更多是为了自回归
            * 图像的位置编码更多是为了理解图像结构

    * 4.21
        * 总结最近关于 VLM 信息流动的调研
            * register token 那篇：
                * DINO 等模型的特征图存在伪影(artifact)，即在背景中与图像无关的 patch 会有较大的 norm。实验(训练分类器)发现这些伪影聚合了 global features；
                * 在 ViT 训练时在序列尾端加上 [Reg] 会改善性能；
            * neo 关于 VLM 可解释性那篇
                * Related works 里详细介绍了 CLIP 的可解释性
                * VLM 是否看见了关键物体？：通过 ablation image tokens 的 embedding 发现可以；
                * VLM image token 信息：给 image token 直接 unembedding
                * **VLM 信息流动：attention block，发现信息流动主要在中层，浅层主要负责全局信息提取**
            * vlm causal tracing 那篇
                * 发现 VLM 偏向于在浅层 MLP 提取信息
                * 图像信息会被转移到图像序列末尾的几个 token 上
            * What's in the image：（1）**text tokens 聚合了全局信息(信息流动：在生成过程中阻断生成内容和 image tokens 之间的注意力)**；(2) 视觉信息的处理集中在中层；(3) 模型能关注到 query 中的特定物体的位置
        * 关于 VLM attenton 的调研
            * 空间推理提高 attention 置信度那篇：证明模型能锁定关键物体，但置信度不一定高。根据生成内容的 logits 来衡量模型的“信心"，从而操纵注意力
            * MLLMs Know Where to Look：证实模型知道看哪儿；通过缩放的方法，把关键区域的部分裁切超分，再接到 image token 末尾
            * DINO 论文头图：ViT [CLS] token 的注意力图
        * 关于 image token 语义：
            * Neel 去年三月份的 post：segmentation map
        * 热力图绘制的方法
            * 基于 attention：如 LVLM-Intrepret，海博师兄的 tool，空间推理 attention 置信度那篇，register token 那篇
            * 基于梯度：saliency map(concept attrntion 那篇里的对比介绍), gradCAM, ...
            * 语义分割
        * 其他
            * Eyes Wide Shut?：罗列了 CLIP 无法解决的九大场景，给了 benchmark
    * 4.22
        * new test
            * 我们的 focus 是自回归模型对空间结构的理解
                * 现在观察到: (1)打乱 ViT 编码后的 image token 序列，对模型的性能几乎没有影响；(2)删去 ViT 全部位置编码，性能很差；(3)删去 LLM 部分的整个序列的位置编码，性能略差；删去图像部分的位置编码，没有影响
                    * (1) ViT 已经给图像加入位置信息了，且这种位置编码根深蒂固影响深远，以至于 LLM 加上错误的位置编码都不会影响；
                    * (2) ViT 负责给图像加入位置信息；
                    * (3) 语言肯定需要位置编码，如果删去 LLM 部分的位置编码，图像有 ViT 之前的位置编码，但语言就啥也不剩了。
                * 现有研究又发现，通过在特定位置上的 attetion block，观察到浅层提取全局信息，中间层提取跟问题相关的目标的信息   
                    * 在打乱 image tokens 之后，是否还会有同样的结论？
                    * 我的猜想是：即使在 LLM 处打乱 image tokens，由于后者经由 ViT 的处理已经包含了位置信息，所以打乱并不会造成影响
                * 那是否可以断定 ViT 已经提取了图像的大部分信息，像相对位置这种信息已经早就处理过了？
                    * 怎么定义和衡量相对位置信息？
                    * !!! 如果真如此，那我们可以做什么？
                        * 更好地理解、解释
                            * 双向自注意力更有利于图像相对位置关系的捕捉，语言模型做的单向自注意力，更多是为了信息的传递
                                * ! 双向注意力增强：在LLM早期层引入受限的双向注意力（如局部窗口注意力），探索是否提升对图像局部结构的利用效率。
                            * !、 ViT 可解释性？(看一下VLM可解释性综述，√VLM可解释性论文，信息流动论文，arXiv)
                            * ~~! 跨层特征相似性分析：计算不同层特征在token顺序扰动前后的相似度（如CKA、SVCCA），定位空间结构信息固化到语义特征的临界层。~~
                        * ! 更好地干预，提性能
                * √ 读论文(What's in the image)
    * **4.23**
        * ! How does the relative positions represented in a VLM? (Do you see it? Visual Geometry in vision language models)
            * 动机
                * 大部分 VLM 是自回归的形式，有些反直觉，想高清模型如何通过逐行扫描的方式理解图像
                * VLM 在 LLM 部分删除 image tokens 的位置编码，即使加上了错误的位置编码，对最终性能影响不大，所以猜测 ViT 编码后的视觉信息包含了鲁棒的位置信息。
                * "Orientation and Direction" 是 CLIP 编码器排名第一的弱点。研究好 VLM 如何提取和利用位置信息至关重要。
            * 怎么定义和衡量相对位置信息？
                * ViT 中的位置信息 (重点是如何表征)
                * VLM 中的位置信息 (重点是如何利用，以及新旧位置编码的相互影响)
                    * 对于某一方位，在一个 batch 上考察其表征。对于某一个样本，先拿到两个物体的 patches (可以利用数据集现有的锚框，也可利用 logit lens)，然后利用这些 patch 的表征提取方位信息
                        * 关于怎么拿到图片中物体的方位信息：利用锚框是精准的；而利用 logit lens 可能还能探究其他的内容，例如 logit lens 何时能够解码出正确的图像信息？利用这个分界点前后的表征拿到的方位信息，是否有所不同？
                        * 方位信息在哪里？
                            * ViT 的卷积核在提取特征时，是否能拿到方位信息？
                                * 调研：卷积核能否拿到方位信息 (够呛，因为 ViT 里面就一层卷积)
                            * !!! ViT 的 transformer 部分会做不同物体 patches 之间的注意力，这个过程中什么时候能够产生位置信息？
                                * 对于一张图片，写不同的文本描述，描述两个物体不同的方位。对于 ViT 每一层的输出，将其分别与这几条描述计算相似度，观察不同方位描述随 ViT 层数的变化（希望有一个上升，其他都下降）
                        * 假设方位信息是线性叠加在表征上的
                            * 虽然 RoPE 是乘性的，但最终转化为注意力分数，是以注意力权重的形式出现的，最终通过 attention 影响了 V，也就是说位置信息通过线性加权的方式融入了 V
                            * 可以利用差分，也可以用其他方法(各种核(38:50)：https://ttv.cn/archives/10280)
                        * 假设位置编码不是简单的线性
                            * 通过逻辑回归等分类器来提取？
                            * 分析位置编码的结构，怎么从表征中复原出位置编码？
                                * RoPE 
        * ! 如何更好地干预，提性能
    * 4.24
        * 打乱图像部分的位置编码(打乱positional ids)
    * 4.25
        * 调研 ViT 和 VLM 对 spatial reasoning, direction information 的研究
        * 调研空间推理数据集
            * VSR
            * What's Up
        * 为了证明 ViT 确实在一步步提取方位信息，我们逐层检查 ViT 的方位表征，看什么时候能够提取方位信息
            * 观察方位信息在哪一层开始拿到
                * 问题
                    * qwen 为了兼容图像和视频，把图像复制了一次
                    * √ qwen vit 进行 batch 编码时，所有图像是拼接在一起的
                    * ViT 后面经过 merger 才能抵达文本，那么对于 ViT 的每一层输出，需要经过 merger 才能由 1284 维转为 3584 维，从而与文本维度对齐
                * 方法
                    * 第一类
                        * image 和 text 的 [cls] 计算相似度
                            * text 表征哪里来？
                        * image 序列 和 text [cls] 计算相似度
                            * text 表征哪里来？
                        * image 序列 和 text 序列 计算相似度
                            * text 表征哪里来？
                        * text 选 embedding 还是 last hidden state? (应该是 last hidden state)
                        * 用相似度，还是方位词接收的注意力？
                    * 第二类
                        * 测 clip 模型
                    * 第三类
                        * √ 直接测 VLM，只不过拿 ViT 不同层的输出经过 merger 变换后的结果
            * 之前研究发现：跟 CLIP ViT 相比，VLM 的 ViT 更能高效地获取这些信息
    * 4.26
        * test direction
            * What's up: 分别在 A 和 B 上测了
            * VSR：打算用二分类(True or False)
            * COCO-spatial 和 GQA-spatial
        * 起名字
            * Reading Images Like Text: Sequential Image Understanding in Autoregressive Vision-Language Models
        * 总结前述想法，进一步猜想：VLM 做 VQA 任务，主要分为两个阶段：
            * 第一阶段：ViT 提取所有图像信息
            * 第二阶段：在文本的引导下，通过注意力机制从图像序列中拿相关信息
        * 下一阶段的计划
            * 针对第一阶段
                * 深入剖析 ViT 是什么时候通过怎样的形式获取和表征图像中物体的方位信息
                    * 可以对比一下裸奔的 CLIP-ViT 和经过 VLM 训练过的 ViT (受论文 https://arxiv.org/pdf/2411.05195 启发)
                    * 比较不同位置编码的效果
                        * 哪个位置编码效果好，为什么好(任务性能好，表征的几何结构好，...)
                        * 如何给位置编码的设计提供 insights？
                    * 分辨率也是重要因素 (受论文 MLLMs Know Where to Look 启发)
                * 深入剖析 LLM 如何利用视觉信息
                    * 现有的研究主要关注从视觉到文本的信息流动
                        * 结论通常是从浅层到深层负责不同任务
                        * 方法大多是 attention block、归因法
                    * 怎么能用某种方法直接或间接地描述出语言模型如何看视觉信息中的方位、空间信息？
                        * 最好能可视化
    * 4.27
        * 测 llava1.6 和 intern2.5-vl (sb)
    * 4.28
        * 测 intern2.5-vl (好鸟)
        * 突然想到，如果在 GQA 上测，结果会不会跟在空间推理数据集上测相同，这样就不能凸显空间推理的主题了
            * 是否可以开阔一下思路：在各式各样的数据集上测，分别观察各种信息分别在哪一层完成提取？
                * ! 仔细读一下 jacob 的两篇论文，里面有关于 CLIP 神经元提取什么样的特征的研究
            * GQA 混杂了各种各样的能力，可能一个问题既有空间推理的考察，也有其他能力的考察
                * √ 读信息流动论文，看他们是怎么区分的
                * √ 读 What's up 论文，看他们是怎么过滤出空间推理问题的
                * √ 是否可以根据问题中是否含有方位词来过滤？/raid_sdd/lyy/dataset/visual-spatial-reasoning/analysis_scripts/rel_meta_category_dict.txt
    * 4.29
        * 解耦 object dectection 和 spatial reasoning 能力，将 GQA 的问题分为带方位词和不带方位词的两类，观察效果。
            * 发现也不能很好地解耦，都是在第15层开始性能有较大的提升。
            * finding：ViT 在处理物体识别(及特征提取)和理解空间方位同时进行
        * 后续安排
            * 试图解耦，看 CLIP 可解释性论文
            * 合理猜想：空间位置信息储存在 attention 模块
                * 传统的卷积神经网络的卷积核有固定大小，在操作上也很像分 patch。CNN 理解空间位置关系是通过滑动卷积，即 patch 与 patch 之间有重叠；而 ViT 的 patch 间是不重叠的，通过 attention 来捕捉相对位置关系
            * ! 现在要探索的问题
                * 问题
                    * 怎么将 VLM 负责空间位置关系探索的部分和物体识别的部分解耦？
                        * CLIP 可解释性论文！
                        * ViT 可解释性 (或许能够作为我们的另一大贡献)
                            * Neel 论文：dogit lens; canonical attention heads
                    * VLM 怎么观察一个完整的物体？一个物体的 patchs 可能不是连续的
                    * VLM 是如何观察两个物体之间的关系的？
                    * ViT 和 LLM 两个阶段分别分析，看视角有何不同
                * 方法
                    * attention block
                    * 除了 attention block，有没有什么更好的方法，最好能将模型的"view"可视化出来
    * 5.6
        * 任务
            * √ 看 CLIP 可解释性论文，看 what 和 layout 怎么体现
            * √ 看 dogit lens 论文，看有没有对 seg map 进行聚类及分割；
            * 思考如何解释 ViT 部分如何观察一个物体，也就是将同一物体的不连续 patch 关联起来？
            * 读位置编码极大值论文
            * 读 learning dynamics 论文
        * 梳理我们的工作
            * 动机
                * 自回归式的 VLM 以逐行扫描的方式观察图像，与人类有很大不同
                * 人脑处理视觉信息有两个回路，分别处理物体识别和方位识别，那这两方面如何在 VLM 中体现？
                * 一系列前置 ablation 实验发现视觉信息的处理主要集中在 ViT 部分，那如何在 ViT 中进行解耦分析，也同时为 ViT 提供更好的可解释性
            * 要解决的问题
                * 1. 物体识别：VLM 怎么观察一个完整的物体？一个物体的 patchs 可能不是连续的，是否在 ViT 的某一层涌现出不连续 patch 之间的关联？（可视化分析）
                    * 物体检测在何处发生和涌现
                        * 从扰动(删层)的角度看什么时候性能涌现
                        * 从 logit lens 角度，观察 seg map
                            * ! 图像是 a bag of words 吗？
                            * ! logit lens 法，用 llm 的哪一层去解码？
                        * 从 attention 的角度，看同一物体 patches 间的相互注意力
                    * 物体识别的两阶段：语义分割和语义消歧
                    
                * 2. 方位检测：VLM 是如何观察两个物体之间的关系的？
                    * 位置信息如何表征
                    * 上下/左右的识别，有不同的 pattern？
                * 3. 模态交互：ViT 和 LLM 两个阶段分别分析，看视角有何不同，也就是说 ViT 和 LLM 如何处理输入得到输出？（比信息流动更细致地进行分析）
            * 可行的方案
                * 对于物体识别，可以借助 logit lens 或直接运用表征来进行聚类和语义分割，看在哪一层涌现
                * 对于方位识别，可能需要从位置编码和注意力模块下手来进行分析
                    * 方位是否可以是线性的，例如"桌子+上方=苹果"
                * 通过微调，比对微调前后空间推理相关结构的变化
    * 5.7
        * 任务
            * 思考如何解释 ViT 部分如何观察一个物体，也就是将同一物体的不连续 patch 关联起来？
            * 读位置编码极大值论文
            * 读 learning dynamics 论文
        * 关于 seg map
            * neel 是把 ViT 的每个 token 经过了 classification MLP
            * neo 是用logit lens，就是直接用 W_{U} 把 img tokens 解码出来了
            * Gandelsman 是用他们的 CLIP decomposition 的方法进行图像分割
            * 我们想要的方法
                * 在 VLM 里面做
                * 不是分类任务(不是 Gandelsman 那样只归因到类别对应的实体上)，而是对整张图像做语义分割
                * 方法
                    * (1) 直接对 hidden states 进行聚类
                        * 聚类还要探索一下
                    * (2) unembedding 法 (基于 neo 论文的发现，以及 geva 的结论 "transformers build predictions by  promoting concepts in vocabulary space")
                        * 算法
                            * 将每个 img token 解码，假设得到 k 种 token
                            * 对 token 进行清洗，减少 k 值
                            * 为 k 种语义 token 赋予 k 种颜色，可视化，得到语义分割图
                        * 一开始想的是直接把 qwen ViT 某一层的输出经过 merger 后，再用 W_{U} 解码。但现在察觉到，之前的实验证明 LLM 阶段 image tokens 可以解码成文本 token 的原因是有 attention 的计算，这使得 image token 拥有了 text token 的成分。而且这种现象还是在 LLM 最后几层出现的，更说明不能直接在 ViT 处解码；
                        * 更进一步，其实 image token 在 LLM 阶段已经没有顺序的性质，而在 ViT 阶段有。所以我们更应该努力探索 ViT 阶段 image tokens 的顺序性；
                        * ! seg 实现起来，可以将 viT 某一层的输出直接送到 LLM 端去推理，看最后 seg map 的效果。这样就是间接证明 ViT 在某一层涌现出物体识别的能力
                            * qwen 会解码出很多中文字符，不太方便聚类以可视化
                    * (3) attention 法
                        * 通过 ViT attention，获取 image token 间的联系
                * 意义(目的)
                    * 能够知晓 ViT 从哪一层开始能够将序列中同一物体不连续的 patches 关联在一起 (Associate discontinuous patches of the same object together in a sequence of image tokens)
    * 5.8
        * Reading Images Like Texts: Sequential Image Understanding in Autoregressive Vision-Language Models
        * 任务
            * seg 实验
                * 动机
                    * 做seg是为了探究ViT是怎么将不连续patch组合起来的，也是为了解释为什么删掉ViT的位置编码后效果会变差。
                    * seg 的目的在于研究物体识别的能力在 ViT 的哪一层涌现出来 
                * (1) 直接对 hidden states 进行聚类
                    * 找一些简单的图片，看同一物体的 token 能否聚类到一起
                    * 现在感觉不太靠谱，因为即使是看起来相同的 patch，在乘以位置编码后也不太相同
                * (2) unembedding 法 (基于 neo 论文的发现，以及 geva 的结论 "transformers build predictions by  promoting concepts in vocabulary space")
                    * 试一下 llava
                    * 想办法对解码出的 text token 种类进行降维
                * (3) ! attention 法
                    * 尝试通过 ViT self-attention，获取 image token 间的联系
                    * 调研一下有没有用 ViT self-attn 做语义分割的
                * 思考
                    * attention 法侧重于研究 ViT 如何通过 位置编码加强的 attention 来实现将序列中同一物体不连续的 patches 关联在一起
                    * unembedding 法侧重于研究 ViT 在哪层开始得到 patches 的语义信息
            * 读位置编码极大值论文
            * 读 learning dynamics 论文
            * 备忘
                * llava1.5 的 vision_tower 用的是 CLIPVisionModel。见 /raid_sdd/lyy/miniconda/envs/mm/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py；
                * llava1.5 会先调用 get_image_features 获得去除 [cls] token 之后的 image embedding，而 qwen2.5-vl 的 ViT 没有 cls token；
    * 5.9
        * 昨天试了 unembedding 法，还是有点用，但是不好 scaling，以及图像还是有点杂乱
            * 借助 GQA 数据集的 bbox 来限定图像中要考察的对象
            * 减少语义分割的类别
        * 昨晚测试了北极熊的 unembedding，惊奇地发现在树干和北极熊之间解码出了"on"，就好像 "bear on the log"!
        * 任务
            * unembedding 法
                * 多测几个简单一些的例子
                * 想办法处理 "in" 和 "on"
                * 想办法处理 "." 和 "\n"
            * 读位置编码极大值论文
            * 读 learning dynamics 论文
        * 新发现
            * 今天测试了小狗坐车和小孩滑雪的图片，发现了一些值得思考的点
                * 解码出的 token 展示了对图像不同层级的理解
                    * 对于摩托车，有: white, window, motor, drive, park。描述了车的颜色、组件，车的名称，车的状态及用途
                    * ! 可以检查一下，看什么时候能解码出代表整个物体的词
                    * ! 可以检查一下，方位词在哪里出现
                    * ! 检查一下同一物体的识别过程(例如识别狗的时候，在浅层会识别成 cat)
    * 5.10
        * 任务
            * obj detection
                * 是否能从 attention 的角度进行聚类？
                    * 思考：模型是如何将同一物体不连续 patches 关联起来的？通过 attention!s
                    * 根据 bbox 选择主要物体，并计算主要物体之间的相似度
                * unembedding解码出的 token 展示了对图像不同层级的理解
                    * 对于摩托车，有: white, window, motor, drive, park。描述了车的颜色、组件，车的名称，车的状态及用途
                    * ! 可以检查一下，看什么时候能解码出代表整个物体的词
                    * ! 可以检查一下，方位词在哪里出现
                    * ! 检查一下同一物体的识别过程(例如识别狗的时候，在浅层会识别成 cat)
            * direction & layout
                * 分别测 left/right 和 on/under，并分析模型如何识别这两种 pattern?(ABAB v.s. AABB)
                    * √ qwen
                    * llava
                    * intern
                * 测一下 logit diff 的变化
            * delete pos embed
                * 测 intern 和 llava1.6 (这俩都是可学习的绝对位置编码)
            * 读论文
                * 读位置编码极大值论文
                * 读 learning dynamics 论文
    * 5.11
        * 任务
            * test_direction_left_vs_on
                * 测 qwen, intern, llava
                * 测 whatsup a and b
            * obj detection
                * 是否能从 attention 的角度进行聚类？
                    * 思考：模型是如何将同一物体不连续 patches 关联起来的？通过 attention!s
                    * 根据 bbox 选择主要物体，并计算主要物体之间的相似度
                * unembedding解码出的 token 展示了对图像不同层级的理解
                    * 对于摩托车，有: white, window, motor, drive, park。描述了车的颜色、组件，车的名称，车的状态及用途
                    * ! 可以检查一下，看什么时候能解码出代表整个物体的词
                    * ! 可以检查一下，方位词在哪里出现
                    * ! 检查一下同一物体的识别过程(例如识别狗的时候，在浅层会识别成 cat)
            * direction & layout
                * 分别测 left/right 和 on/under，并分析模型如何识别这两种 pattern?(ABAB v.s. AABB)
                    * √ qwen
                    * llava
                    * intern
                * 测一下 logit diff 的变化
            * delete pos embed
                * 测 intern 和 llava1.6 (这俩都是可学习的绝对位置编码)
            * 读论文
                * 读位置编码极大值论文
                * 读 learning dynamics 论文
        * 思考
            * left v.s. on 实验符合预期，下面就是研究模型如何识别两种 pattern
            * 完成了 attention 法的代码，效果跟预期不符
                * ! 猜想：物体识别分为两个阶段：语义分割和语义消歧
                * 代码写错了，明天需要继续探究 attention 法和 sim 法
            * 下面更进一步研究 logits lens 的细节
    * 5.12
        * √ 昨天睡前改好了寻找 bbox 内的 patches 的代码
        * 画 sim 曲线，一点点符合预期，接下来：
            * 搭配 attn 一起探究
            * 研究 logit lens，力求分离：语义分割和语义消歧
        * findings
            * sim 曲线显示模型在早期(5层左右)开始学会语义分割，自相似度在中间层达到顶峰，后面相似度又降低
                * 典型
                    * bear: rock, nose
                    * ski: ski, clothes
                * √ 后面下降的原因
                    * 可能是完成了语义分割，然后通过关注其他物体(上下文)来完成消歧
            * attn 总体呈现中间层高，两边低的特性
                * √ 何意味？

        * 思考
            * √ 除了画某个 obj 对自己及其他物体的注意力变化，还可以计算差值
    * ！！！5.13
        * 任务
            * √ unembedding 法用哪一层解码？
            * unembedding解码出的 token 展示了对图像不同层级的理解
                * 对于摩托车，有: white, window, motor, drive, park。描述了车的颜色、组件，车的名称，车的状态及用途
                * ! 可以检查一下，看什么时候能解码出代表整个物体的词
                * ! 可以检查一下，方位词在哪里出现
                * ! 检查一下同一物体的识别过程(例如识别狗的时候，在浅层会识别成 cat)
            * direction & layout
                * 分别测 left/right 和 on/under，并分析模型如何识别这两种 pattern?(ABAB v.s. AABB)
                    * √ qwen
                    * llava
                    * intern
                * 测一下 logit diff 的变化
            * delete pos embed
                * 测 intern 和 llava1.6 (这俩都是可学习的绝对位置编码)
            * findings
                * 突然察觉到：关系为 on 的两个物体，会有很多交叠！不是简单的一个在上一个在下
                    * 左右还是难分难解，可以用原来的猜想继续解释
                    * 怎么处理上下？下还好一点，因为在下方，往往不会与物体有很多交叠
                    * ! 回归本质：单纯的双向 self-attention 无法捕捉位置信息，需要位置编码帮助
                        * 加了位置编码的 ViT 就像二维的 BERT
                        * 之前 eyes wide shut 说语义相反的两张图片，其表征可能很相似，但他用的是 CLIP-ViT-L-14。qwen 是否会出现 clip blind pairs 的现象呢？
                        * llava 用的 CLIP-ViT-L-14 的位置编码是绝对位置编码，影响可能比较弱，而 qwen 用的 ViT 用的是 RoPE，位置编码更强。但也不能这么说，毕竟 intern ViT 用的也是绝对位置编码。
                        * 位置信息到底是怎样表征的呢？？？
                            * 可能 intern 图文对齐训练得好
                            * ! 可能的实验：就用 VSR 的图片，因为变量控制得好，只有一个物体的位置发生了变化
            * 读论文
                * 读位置编码极大值论文
                * 读 learning dynamics 论文
                * [! Learning Visual Composition through Improved Semantic Guidance](https://arxiv.org/pdf/2412.15396)
            * 备忘
                * 关于 seg_with_unembedding 的代码
                    * bear: 不用 text_tokens_copy
                    * ski: 用
    * 5.14
        * 思考
            * VLM 空间感知，位置嵌入
            * logit lens 物体边缘会解码出 token，内部为无意义 token
                * 现象不够典型
            * 画语义消歧图：属性词/近义词和代表性词的变化
    * 5.15
        * 继续 seg，发现由局部涌现到整体上一个普遍规律，例如在鸟头里的 patches，从 mouth, eyes 到最后统一变成 bird
            * finding: hidden attributions：比如鸟嘴对应的 pacthes，最终会由 bird 占主导，但也包含了 mouth/beak 的语义
        * 空间感知
            * 可以训练一个分类器
                * 输入：两个物体整体的图像区域；小物体(目标物体)的区域
                * 输出：方位标签
    * 5.16
        * √ 测 intern 在各个方向上的 logits 变化
        * 发愁，不知道怎么策测量 VLM 的空间表征
            * mutual
            * 差分
            * pos embed
            * 想要找到一种测量方法和指标，从 VLM 的隐层表征出发，衡量模型 visual geometry 的好坏
        * 测：
            * 打乱 llava 的 image tokens
    * 5.17
        * 空间感知不好的原因
            * resize 变形
            * qwen merger 干扰了位置编码
            * clip 本身不行
    * 5.19
        * 聚类与降维
            * 方向向量：池化的 patches 相减
            * 灵感来源：
                * [位置细胞，网格细胞](https://mp.weixin.qq.com/s?__biz=MzI5MDQzNjY2OA==&mid=2247556835&idx=1&sn=624107ba55f05749d844288f663540a0&chksm=ed1637725c92a9cf65f742a494ec8149bb971b7647ba6fddd069cf6f0ccb62512c7bdf427ad7&scene=27)
                * [纳家勇治课题组在PLOS Biology上发表论文，揭示空间记忆提取的神经机制](http://www.psy.pku.edu.cn/xwzx/xyxw/372402.htm)
            * 开始对方向向量进行降维(t-SNE)和聚类(k-means)
                * √ 是否标准化
                * √ t-sne v.s. k-means
                * √ llava1.5 v.s. qwen
                * 如果假设方向特征是线性的话，是不是要做PCA
        * 判断四个方向的向量是否正交
        * 干预实验
            * 是否能利用刚才的实验，对模型进行干预呢？
                * 可视化：
            * 通过相减能得到方向信息，那在实际推理过程中，是否体现了这种相减性？
            * (是否还能发现代表方向的神经元？)
    * 5.20
        * 干预实验
            * 干预的目标是让 qwen 效果变差，也可以是让 llava 的效果变好
        * 聚类实验
            * 发现了一个大 bug：get_relation_representations 函数里，实际上只考察了一条样本，形成不同点的原因是 object bboxes 没有问题，相当于在一条样本里取所有样本的不同的 bboxes。
            * 修复了 bug
                * llava 的聚类结果存在了 "/raid_sdd/lyy/Interpretability/lyy/mm/eval/WhatsUp/results/test_vit_directions_relation/llava1_5_7b/good"，但随机选的 patches 的数量是所有 patches 除以2。更好的结果在标准路径下
    * 5.21
        * 干预实验完成
        * 检查方向向量是否与文本表征对齐
            * 对数据集里的每一条样本，输入为"图片+正确描述"。计算描述中的方位词与方位向量以及其他物体和背景的相似度
                * 可以先看某一条样本
    * 5.22
        * 表征对齐效果不是很好
            * 文本跟方向向量的相似度最低
            * 其实没问题，方向向量是模型在看到"behind"和nucelus后自己"想到的"，而非一定要与 behind 对齐
            * 可以先从 attention 入手
                * attention 也是屎
        * 尝试对 llava 进行积极的干预
            * 失败，性能会下降，甚至出现乱码。两种可能：
                * llava 的方向表征(位置编码)很差
                * llava 的方向不能用方向向量来衡量
                    * 干预实验本身就是想证明方向向量表征的好坏和性能的因果性，但是如果 llava 的表征很  烂，那就很难通过一般的干预来证明因果性，但也不能否认因果性...
                    * 一个可能的方法是，想办法调整 llava 的表征，使得其方向向量的表征好看一些，然后看性能有没有提升
                    * qwen 方向向量良好的正交性可能来源于先天正交的二维位置编码
    * 5.23
        * 在 intern 上做最近的实验，比较 一维可学习的绝对位置编码 和 二维RoPE 的区别
            * 要想区分上下左右，至少需要两个维度
            * 二维位置编码，本身就考虑了两个坐标，把 x, y 两个维度的位置信息融入表征，所以方向向量的差值相当于二维坐标系里向量的差值，自然能反映上下左右
                * 在推理过程中，模型锁定 nucleus 的位置后，可能会根据方向向量来检索对应位置的 satellite
            * 而一维位置编码，只有一个维度，因此在表征上下左右的方向时，会依赖图像的尺寸(width)信息，因为我们可以也只能通过某个 patch 的 id 和图像的 width 来推算图像的二维坐标。而对于模型而言，这种二维坐标的推算是否存在？
                * 我倾向于不存在。因为与 RoPE 不同，一维位置编码是可学习的，也就是说 336*336 / 448*448 的图像里的每一个 patch id 对于模型来说都是与众不同的。由于 patch id 个数有限且固定，模型很可能会学习出不同 patch id 之间的位置关系，例如 id24 可能在 id 53 的左边。在这种情况下，通过相减得到的方向向量，似乎没有直观上的意义(毕竟不是在坐标系里的向量)。因此，需要寻找一种新的方法来度量一维位置编码得到的方向信息。
        * 删 llava 位置编码
        * 测 llava, intern, qwen 在 Whatsup A/B 上的性能
        * 测乘性的方向向量 
        * qwen 的上下左右正交性是否与三角函数有关？
    * 5.24
        * 删 internvl llm 部分的位置编码
        * 绝对位置编码
            * 测位置信息在 patches 中的比例，观察衰减
                * 对比 llava 和 internvl
                * 思考衰减快的原因
            * √ 一维位置编码的 visual geometry
        * RoPE
            * qwen ViT 不同层的方向向量聚类，观察演化过程
            * 测 qwen2-vl
    * 5.25
        * 绝对位置编码
            * 发现 internvl 比 llava 衰减还快，说明这招行不通
            * 想办法解释：llava 逊于 intern 的原因
        * RoPE
            * qwen ViT 不同层的方向向量聚类，观察演化过程
            * 相减得到的方向向量是否能揭示方向信息？？
                 * 测 qwen2-vl, qwen2.5-vl-3b。首先发现4个聚类指标，都是在起始和终止位置有较大变化，而在中间层比较平缓
                 * 但是聚类的好坏不能反映性能的好坏。这也合理，因为某层的输出只是物体本身的语义表征和位置信息的融合，这点不同的模型之间都差不多，所以聚类结果也都大差不差，跟模型的性能好坏没有多大关系。后面的工作就是探究这种相对位置信息是如何被利用的，怎么衡量利用度好坏，以及跟最终性能的关系。
                 * 我还是相信相减能够反映出一些位置关系的。在三个模型上，都反映出了左右/上下分别组成共线对，而左和上则正交的现象。并且干预实验也证实了这一点
    * ！！！5.26
        * 在 llm 推理前，删掉两个物体的其中一个，看对性能的影响
            * 分别测试绝对位置编码和相对位置编码，看运行机理是否一样
            * 这个实验的目的是探究位置编码是如何表征的。因为位置编码的目的是加入位置信息，而最终的直接作用区域就是注意力模块，而注意力模块又是唯一能够实现 token 间通信的模块，所以位置编码会将其他 token 的语义信息和位置信息融入到某个 token。为了验证这点，我们就可以抹掉两个物体中的其中一个，看当只有一个物体时，模型能否从这一个物体的表征中提取到两个物体的语义信息和位置信息，并成功完成预测。
            * 实验结果证明，模型能够在目标物体不在场的情况下完成预测(从其他 vision tokens 中提取相关信息)
                * 事实上，token 压缩等工作也证实了这一点，即模型的信息最终会汇聚在 last token 上。这很合理，因此 VLM 最终要做的是文本生成，而下词预测所需的信息全部来源于 last hidden state。所以图像信息回努力汇聚到 last token 上。
                * 一个 image token 汇聚了众多 image token 的语义信息和位置信息，那如何获取图像表征中的方位信息？
                    * 一维绝对位置编码：每一个位置都有其专属的位置编码。当问"桌子下面是什么？"时，模型会先关注到桌子的 patches，记录下桌子的位置信息，然后根据该位置信息推导出桌子下面的物体的 pacthes 对应的位置，并去查看那个位置对应的语义信息是什么；
                    * 二维 RoPE：处理可变尺寸的图像，所以需要同时获得横坐标和纵坐标的信息，从而推断出某个 patch 所在的位置。
            * 位置信息的表征(visual geometry)
                * 一维：通过对位置编码聚类来可视化
                * 二维：对于一个物体，左右的区别在于其表征中包含的其他物体的相对位置
                    * 可以对同一个物体的表征进行聚类，如果对于不同的模型，聚类结果都很好，那就说明空间推理的 bottleneck 不在于位置信息的表征，而在于位置信息的利用。
                        * qwen2.5-vl v.s. qwen2-vl, bowl v.s. candle
                    * 那之前做的方向向量还有用吗？有用，因为对于某两个方位(left v.s. right)，两个物体表征之差的公共项非常之多，而差项则反映了位置关系的不同，可以作为方向的一种衡量指标，即之前提出的"方向向量"。基于方向向量可以对空间推理进行干预。
            * 位置信息的利用
                * 两类问题：
                    * 桌子在图片的哪边？这个问题需要以整幅图像的中心为原点，建立全局坐标系。
                    * 桌子的下面是什么？这个问题需要以桌子为原点，建立相对坐标系。
    * ！！！5.30
        * 为什么左和右是共线(反向)的，而和上是正交的？
        * clip blind pairs 的结论是否正确？VLM 的 bottleneck 是否在 ViT 的图像编码能力上？
        * 备忘
            * 获取 ViT 输出，before connector
                * check_relation_pair_similarity
            * 获取 ViT 输出，after connector
                * get_relation_representations
            * 获取 ViT 输出，进行干预，再传给 LLM 进行推理
                * intervene_in_spatial_reasoning
            * 删除 ViT 某层及之后的层
                * test_vit_direction_vlm
            * 获取 LLM logits
                * test_vit_direction_left_vs_on
            * 获取 object patch 在 ViT output 中的 ids
            * 获取 object patch 在LLM input 中的 ids
                * check_direction_language_alignment
    * 6.3
        * 下载了 MMVP 数据集，看了一下质量，不适合标注，打算作为测试集用
        * 下一步
            * 开启第三阶段的探索
                * 删除 Llava 和 Intern 在 LLM 部分的位置编码，观察性能变化
    * 6.4
        * 之前对 RoPE 提取位置信息的手段，在一维位置编码上是否适用？
            * erase object：已经测过了
            * 从单个物体内提取位置信息：没测
            * 从两个物体内提取方位向量：已经测过了
        * 聚类结果能说明什么呢？
            * 首先，我们只是提出了一种提取和可视化方位信息的方法，而这种方法与模型内部的运作机理是不一样的
            * 但是，我们方法能够还原出模型内部对方位信息的表征，从而证明 VLM 的视觉编码器确实具有表征方位信息的能力
            * 但是，聚类的指标能反映方位信息表征的好坏吗？好像并不能。
            * 不过话说回来，我们的方法无非是（1）用物体的表征聚类 （2）用两个物体的表征之差聚类。没有做过多的主观操作
            * 此外，聚类结果与模型的空间推理性能没有相关性，也就是说方位信息表征好，不一定就代表最终性能好，还要看 LLM
            * 此外，t-SNE 效果在所有模型上就都很好啊。是不是说明聚类的数据本身就是线性不可分的，用 PCA 反而不合适。
            * 问题
                * 这两个方法是否适用于一维绝对位置编码？
                * 这两个方法能否反映视觉编码器的好坏？
                * 聚类的目标是什么？
                    * 可视化方位信息的表征，证明 ViT 能够区分不同的方位
        * 下一步
            * 开启第三阶段的探索
                * 删除 Llava 和 Intern 在 LLM 部分的位置编码，观察性能变化
    * 6.5
        * test_pos
            * delete pos embed (vit / llm):  llava1.5, internvl2.5
            * delet pos embed (image patches in llm)
        * 之前的论文 (ViT + LLM > ViT) 说 VLM 能够发现 ViT 看不到的细节，也就是说 LLM 的好。Eyes wide shut 那篇论文说的是 VLM 的 bottleneck 在于 ViT 捕捉不到细节。
            * 首先，ViT + LLM > ViT 其实也说明了 ViT 已经拿到了足够细节的表征，只是说纯靠 CLIP 的模态间余弦相似度难以完成任务，但至少视觉编码器拿到了相对较好的表征，这点可以反驳 Eyes wide shut；
            * 其次，我们的工作要说的是 LLM 还不能很好地利用 CLIP 的视觉表征，也就是说 bottleneck 可能在 LLM，或是整个模型架构。这点其实是在反驳 ViT + LLM > ViT。
            而我们的工作是要证明 ViT 拿到了
        * 下一步
            * 第二阶段的延伸与补充
                * processing dynamics
                    * 位置编码是否是对方位信息表征的一种保持？并不是像物体识别那样逐层深入，而是一开始就有，只不过后面需要保持。如果真这样的话，可以安排一个实验，就是删掉第一层之后的位置编码，看方位信息表征的瓦解
                    * 测 llava 和 intern (test_vit_directions_relation_layerwise)
                    * 测试指标是聚类指标以及正交性，并考虑正交性是用整个物体 pooled 的结果计算，还是一部分
                * 一维绝对位置编码 和 二维旋转位置编码 的区别在哪里？
                    * 删 llava 的 LLM 部分的位置编码还是会导致性能下降
                * ViT 的视觉表征为何能抵抗 shuffle 等操作？
            * 开启第三阶段的探索
                * 位置信息的利用
                    * 两类问题：
                        * 桌子在图片的哪边？这个问题需要以整幅图像的中心为原点，建立全局坐标系。
                        * 桌子的下面是什么？这个问题需要以桌子为原点，建立相对坐标系。
                * 比信息流动更细粒度的研究
                    * 文本于视觉到底怎么交互
                * 
    * 6.6
        * 位置编码在intern和llava里是否会衰减？实验表示如果从最开始就删去llava的位置编码，性能会减少，但不至于减没。如何衡量这种衰减？为什么衰减后，模型还能很好地识别位置关系？通过给绝对位置编码聚类，我们看到位置编码确实学到了行和列的信息，那如果在llava每层都加上位置编码，会怎样？
        * √ 怎么可视化位置信息的动态加工过程？qwen的每一层的聚类效果都差不多，那如果只在第一层加rope，那后面会是什么样的呢？
        * √ 删去位置信息会影响物体识别吗？可以试一试删去llava的位置信息，然后看北极熊的变化。
    * 6.7
        * 逐层分析方位表征
            * 为什么表征聚类效果在第31层暴涨？
            * 曲线的起伏跟模型结构有关系吗？（window attention 这种）
            * 不加位置编码，聚类结果怎样？
        * 位置编码在intern和llava里是否会衰减？实验表示如果从最开始就删去llava的位置编码，性能会减少，但不至于减没。如何衡量这种衰减？为什么衰减后，模型还能很好地识别位置关系？通过给绝对位置编码聚类，我们看到位置编码确实学到了行和列的信息，那如果在llava每层都加上位置编码，会怎样？
    * 6.8
        * 删除 LLM 的图像部分的位置编码
        * 逐层分析方位表征
            * 为什么表征聚类效果在第31层暴涨？
            * 曲线的起伏跟模型结构有关系吗？（window attention 这种）
        * 位置编码在intern和llava里是否会衰减？实验表示如果从最开始就删去llava的位置编码，性能会减少，但不至于减没。如何衡量这种衰减？为什么衰减后，模型还能很好地识别位置关系？通过给绝对位置编码聚类，我们看到位置编码确实学到了行和列的信息，那如果在llava每层都加上位置编码，会怎样？
            * 给 llava 每层加上位置编码，发现性能维持在 0.600
    * 6.9
        * √ 测删所有位置编码在 Whatsup B 上的最终性能，用于补充附表
        * 删位置编码影响物体识别吗？
            *  测删所有位置编码在 GQA 上的最终性能
            *  测删所有位置编码在 gqa_spatial 上的最终性能
            *  测删所有位置编码在 gqa_no_spatial 上的最终性能
        * 逐层分析方位表征
            * ！为什么表征聚类效果在第31层暴涨？
            * ！曲线的起伏跟模型结构有关系吗？（window attention 这种）
        * 位置编码在intern和llava里是否会衰减？如何衡量这种衰减？为什么衰减后，模型还能很好地识别位置关系？如果在llava每层都加上位置编码，会怎样？
            * 给 llava 每层加上位置编码，发现性能维持在 0.600
            * ! 在 Whatsup 上测呢？
                * 也是没有好结果
    * 6.10
        * √√ 再试一下 seg_with_cluster
        * ~~sigclip~~
        * ？ViT 的视觉表征为何能抵抗 shuffle 等操作？
        * ？逐层分析方位表征
            * ！为什么表征聚类效果在第31层暴涨？进一步验证：删掉最后一层，并打乱 vision tokens 的顺序，看性能下降的程度
                * 效果跟之前的 test_vit_direction 基本一样，看不出什么名堂，可能需要其他的方法
            * √√ 曲线的起伏跟模型结构有关系吗？（window attention 这种）
        * 开启第三阶段的探索
            * 位置信息的利用
                * 两类问题：
                    * 桌子在图片的哪边？这个问题需要以整幅图像的中心为原点，建立全局坐标系。
                    * 桌子的下面是什么？这个问题需要以桌子为原点，建立相对坐标系。
            * 比信息流动更细粒度的研究
                * 文本于视觉到底怎么交互
    * 6.11
        * ？ViT 的视觉表征为何能抵抗 shuffle 等操作？
        * ？逐层分析方位表征
            * ！为什么表征聚类效果在第31层暴涨？进一步验证：删掉最后一层，并打乱 vision tokens 的顺序，看性能下降的程度
                * 效果跟之前的 test_vit_direction 基本一样，看不出什么名堂，可能需要其他的方法
            * internvl 的聚类效果随层深下降
                * √ 可能 internvit 对图像进行了分块，块与块之间在 ViT 阶段没有进行交互，所以还要测一下 llava
                * 其实可以验证一下，intern 的 ViT 是否也存在位置信息的衰减，这样就能说明绝对位置编码存在衰减的问题
                    * ！用 thumbnail 进行聚类
                        * one object
                        * direction vector
                    * ！预实验-InternVL
                        * 原始性能
                        * 删 ViT 位置编码
                        * 打乱 intern LLM 处的图像部分的位置编码，看对性能是否有影响（希望影响大一些，这样更符合 intern 的结构，毕竟在 ViT 阶段是子图(tile)处理，子图间没有交互）
                    * ！重新测 intern_thumbnail 的聚类效果
                        * one object
                        * direction vector
        * 开启第三阶段的探索
            * 位置信息的利用
                * 两类问题：
                    * 桌子在图片的哪边？这个问题需要以整幅图像的中心为原点，建立全局坐标系。
                    * 桌子的下面是什么？这个问题需要以桌子为原点，建立相对坐标系。
            * 比信息流动更细粒度的研究
                * 文本于视觉到底怎么交互
    * 6.12（组会）
        * 理论的应用
        * 如果只有实验，还是停留在黑盒解释阶段。最好先有理论，再通过实验验证。
        * 除了四个位置关系，还可以探究其他更复杂的位置信息。从而试图找到 RoPE 可能存在的问题。
        * 尽快进入第三阶段
    * 6.16
        * 正交性与共线性
        * 模态间交互
        * 实际应用
            * 方位辨别 / 物体识别的能力到底在 clip 还是在 llm，需要通过模态间交互的实验来验证
        * 更难的方位辨别
            * VSR
            * on v.s. behind
    * ！！！6.17/18
        * 为什么 LLM 给图像加的位置编码没有用，且 ViT 输出的表征能抵抗新的位置编码？
        * 视觉语言如何交互？  
            * 物体识别
                * 任务："What's in the image?"。拿 ViT 不同层处理得到的表征进行测试，看 VLM 是否都能输出正常的句子？如果能生成正常的句子，识别出的内容是否随层深变得更精确？
                * 如果删掉 vision tokens，会有怎样的变化？
            * 空间感知
                * 模型是怎么从 ViT 的表征里知道上下左右的？
                    * 分析：
                        * 这个阶段的图像序列是不需要位置信息的，也就是说位置相关的信息已经存在于表征里，仅待提取。
                        * RoPE 本质上还是改变了注意力权重分数，从而改变了一个 patch 内所含的其他物体的表征的比例。无论是对图像还是对文本。既然 LLM 阶段不用位置编码，说明 LLM 可以自行分析 patch 中所含其他物体的表征的比例来判断其位置。
                        * LLM 阶段的注意力是 causal 的！图像不再是 ViT 那样可以看到四周，而是只能按序列的因果性来注意。这本身有悖于图像这一二维的模态。这是否是 LLM 阶段图像位置编码失灵的原因？
                        * hongxu 师兄也提到了，位置编码不是必要的，causal 本身就携带了位置信息
                    * 问题：
                        1. satellite 在左和在右，attention pattern 一样吗？需要实验验证一下：
                            * RoPE：理论推导出来是不一样，可以做实验，画图：横轴是 RoPE 划分的维度的组号，比较前后左右的区别
                            * 绝对位置编码：再做一些理论推导，也检查一下 attention
                        2. √ causal attention 是否会影响空间感知？
                            * 之前的 attention block 没有 block vision tokens 之间的注意力。如果去掉 vision tokens 之间的注意力，是否会有影响？
                                * 相当于在 LLM 阶段不再用 attention 模块处理图像（但是还有 MLP），仅让文本从 ViT 图像表征中提取内容。
                            * 假如把 attention block 改成双向注意力，又会有什么影响？causal attention 是否会对某个方向的识别造成影响？换句话说，causal attention 在空间感知过程中是否会偏袒某个方向？
                            * √ LLM 阶段的 causal attention 是否与 ViT 的双向注意力相矛盾？
                        3. 怎么从单个 patch 中分离出各个物体表征的占比？
                            * 是不是需要训一个简单的分类器？
                        4. 模型根据什么来判断方向？
                            * 单个 patch v.s. 多个 patches
                                * 删 vision token，看看只靠一个 vision token 能否完成任务。
                            * patch 里面的 attention pattern（RoPE） v.s. patch 里面的位置表征（绝对）
                            * 靠绝对位置编码？
                                * 读论文：[Probing the Role of Positional Information in Vision-Language Models](https://arxiv.org/pdf/2305.10046)
                            * 靠 attention pattern？
                                * 实验：绘制 ViT attention pattern 随层深变化的过程，纵轴为 attention 的平稳性（相对于上一层的变化率），看 attention pattern 是否会收敛到一个固定的 pattern，并观察不同方位对应的 attention pattern 的区别。
                        5. obj info 和 pos info 的提取过程？
                    * 一些思考
                        * 空间位置关系信息不同于一般的肉眼可见的特征，后者可以直接通过卷积核捕捉和表征。而空间位置关系这一特征涉及到两个物体，也就是多个 vision tokens 之间的交互。是比一般特征更高一层级的抽象特征。
        * RoPE 是否有缺陷？绝对位置编码是否有缺陷？怎么改进？
        * faithfulness of interpretability
        * zzy 提的意见：connector 是根据 ViT 倒数第二层来训练的，直接拿前面的 ViT 层经过 connector，有点不合理
     
    * 6.23
        * LLM 阶段图像改成双向注意力
            * √ qwen2_vl
            * √ qwen2.5_vl：有个滑动窗口注意力，比较难搞
            * √ llava1.5
    * 6.24
        * LLM 阶段图像改成双向注意力
            * 昨天搞错了，qwen2-vl 系列的模型都有滑动窗口
            * 改了以后，效果有很大提升
            * ！为什么改动之后性能会变好？
                * 对图像完成了进一步处理
                * 更有利于图文交互？
            * ！在更多数据集上测
                * whatsup_b: 似乎没什么提升
                * vsr: 
            * ！该方法与 seg_with_unembeddings 结合，看看有没有新的发现
                * 绘制曲线图，看在 LLM 哪一层解码出的关键字最多，结果发现跟原模型几乎没有区别。其实也合理，因为之所以能用 lm_head 从 vision tokens 中解码出 text token，是因为 LLM 阶段注意力的作用。改变图像部分注意力的方向，并没有提升图文交互的效率
                * 那为什么双向注意力能够提升性能呢？
                    * 原本的单向注意力导致图像序列中只有最后一个 vision token 才能看到整幅图像，即只有 the last vision token 包含了所有 vision tokens 的内容。文本侧可能只能依靠这一个 token 来做预测。这种情况下，图像序列是畸形的。
                    * 而双向注意力使得每个 vision token 都能包含其他 vision tokens 的信息，因此整个图像序列的平均信息丰富度就提高了。
        * ！ViT attention 按维度分组研究
            * qwen 对 rope 的实现方式跟我想的不太一样。我之前是按知乎的说法，01一组，23一组，...；而 qwen 是 1-65一组，2-66一组，...。所以这样下来，也就没有区分低维和高维的必要了，而是只需要关注组号（组号小的组计算得到的注意力权重可能是正的，组号大的得出来可能是负的）。
            * 所以我们要做的是：首先拿到 t, x, y 这三个信息对应的 dimension groups。然后分别统计 satellite 对 nucleus 的注意力分数里这三个信息对应的权重，以及每个信息的 dimension groups 的分数的正负性。
        * ViT / LLM attention 演化过程
    * 6.25
        * LLM 阶段图像改成双向注意力
            * cnm，没用，是 GQA 测试集选成 train 了。
            * 能不能两种架构分别在 GQA 训练集上微调，然后比较性能好坏？
        * block 图像注意力
        * ！ViT attention 按维度分组研究
            * 可视化
            * 分析 MRoPE 为什么不起作用
    * 6.26
        * LLM 阶段图像改成双向注意力
            * cnm，没用，是 GQA 测试集选成 train 了。
            * 能不能两种架构分别在 GQA 训练集上微调，然后比较性能好坏？
        * block 图像注意力
        * ！ViT attention 按维度分组研究
            * 可视化
                * 目前还不太顺利
                * 是否可以计算累积的注意力分数？不太好搞
                * 艹，直接 head_dim // 2 不对，head_dim 里位置信息的分配是 (x, y, x, y)，4*20=80
                    * 改了以后，确实能分离 h 和 w
                    * 使用 s->n 的全部注意力分数的平均，效果不好：h/w 分离，但是 left 和 right 没有分开
                    * 试一下 object center 的 patch 的分数？即 s_center -> n _center
                    * 左和右一起归一化
                * 敏感性分数
                    * 梯度
                    * logit lens？即 (v_{left} - v_{right}) 与 satellite 的余弦相似度
            * 分析 MRoPE 为什么不起作用
        * 物体识别可解释性结论，是否可以帮助幻觉的研究？
            * 工作的意义和应用：幻觉机理分析和解决；提升物体识别和空间感知能力
    * 6.27
        * LLM 阶段图像改成双向注意力
            * 测 internvl
            * 能不能两种架构分别在 GQA 训练集上微调，然后比较性能好坏？
        * ！block 图像注意力
            * LLM 阶段图像注意力是否是多余的，特别是对于 llava 来说，新加的位置编码其实是干扰项
        * ！ViT attention 按维度分组研究
            * 可视化
                * 是否可以计算累积的注意力分数？不太好搞
                * ! only center
                * ！怎么更好地比较 h 和 w 注意力分数的大小？
                * ！敏感性分数：证明 attention 的正确性
                    * 梯度
                    * logit lens？即 (v_{left} - v_{right}) 与 satellite 的余弦相似度
            * 分析 MRoPE 为什么不起作用
    * 6.29/30
        * ！ViT attention 按维度分组研究
            * ！累积注意力
            * 对于前后的关系，去掉不好的样本
        * 物体识别可解释性结论，是否可以帮助幻觉的研究？
            * 工作的意义和应用：幻觉机理分析和解决；提升物体识别和空间感知能力
            * ！想法1：用 obj detection 的结论做应用
                * 之前的实验是删 ViT layers，后面还是老老实实经过每一层 LLM layer
                * 现在我们不去动 ViT，而是用 lm_head 依次解码 LLM layer 的表征
                * 选定 LLM 第 i 层，如果解码出的 text token 是标点符号/空格/数字，则删掉
                    * 有一点效果
                    * 幻觉评测
                        * 感觉 pope 数据很适合 object detection 这一部分的研究 
        * 待办
            * qwen seg map
            * 测 llava 在 pope 三个数据集上的表现，以及干预后的表现
            * 去重实验
            * 测 qwen 干预实验
    * 7.1
        * √ 测试 qwen seg map 效果
        * 三种干预
            * 从指定层开始 token truncation
                * 再测一下
            * 将 last hidden states 替换为指定层输出(image info density)
            * turnback
                * debug
                * 效果不好
            * dedup？
        * 尝试在 qwen 上做 token compression 的实验
        * object detection
            * 第三阶段的研究：可分为物体识别和空间感知中的模态交互
                * 其中物体识别任务可以直接选取 pope 数据集
            * seg map 的方法可以直接告诉我们模型究竟看到了什么
                * 能告诉我们模型 fail 的原因，是没有感知到，还是模态交互不行
                    * 一般都能感知到，例如 van，但模型看成了 bus
                    * 但也有感知不到的，例如消防栓
                * 能查出诸如pope这类数据集存在的严重问题。例如: /raid_sdd/lyy/Interpretability/lyy/mm/test_figs/pope/bad_case/llava_pope_popular_Is there a tv in the image?_yes/4326.png
        * 完成了 coco 数据集的处理和 CHAIR 指标的计算。
    * 7.2/3/4
        * 目前的干预：
            1. 在25层删无意义token(待测)；
            2. 根据信息密度筛选层(原始性能差)；
            3. 在25层删无意义token(三层)，然后重新来过(原始性能好，但幻觉差)；
            4. 去冗余：游程编码。
                * ！我们的方法：training-free，instruction-agnostic -- only look at the image
                * 发现空白的区域通常对应大片的单色区域，；例如天空、地面、水面等
                * 所有 patch 均游程编码
                * 只对无意义符号游程编码，meaning pooling
                * 只对无意义符号游程编码, select leftmost
                * 多测几个 layer_id？或者说，根据信息密度自适应地选取层
        * 检查模型究竟能看到多少
            * 解耦视觉编码和模态交互
            * 仅仅让 VLM 回答问题，不是很靠谱，不稳定
            * 方法：对问题中的每一个实体，取同义词集合作为实体名。检查解码出的 token 是否等于实体名，若不等于，再分词，并看解码出的 token 是否等于实体名的第一个 token。
            * 目前超不过直接 prompt vlm
        * 待办
            * √ seg: m5 GQA; seg1: m5 pope2
            * √ 测一下 useful tokens 也参与游程编码的方案
            * √ 尝试删去一些字符：
                * 空格
                * 句号，其他标点
            * 测 llava
                * random select，发现有效果
                    * mme√, poper random√, pope popular√, pope adver√, CHAIR
            * 测 qwen
                * all tokens
                * all tokens, del puncs
                * all tokens, del puns & digits
                * mean pooling v.s. random select
            * 总结，游程编码的方案有：
                * all tokens, mean pooling
                * all tokens, random select
                * all tokens, del puncs, random select
                * all tokens top3, del puncs, random select, 
    * 7.5
        * 备忘
            * seg: qwen, method5.2, pope random
            * seg1: qwen, method5.2, pope popular
            * pos:
            * pos1: 
        * token compression
            * 是否考虑测一下 llava bench in the wild?
        * 梳理
            * 物体识别
                * 两阶段：识别和消歧
                    * 方法：相似度和注意力两个指标
                * ViT 处理过程可视化
                    * 聚类
                    * seg map
                        * 局部特征 -> 整体特征               
            * 空间感知
                * 为什么 ViT 的图像编码尤其是位置编码很鲁棒？
                    * 可学习的一维绝对位置编码
                        * 可视化
                    * 二维 RoPE
                        * 位置信息来源于哪里？注意力分数。RoPE 作用于注意力模块，通关改变注意力分数，改变一个 patch 里该位置和其他位置的 patch 的比例，从而达到区分方向的目的？
                            * 检验(erase)：一个 patch 聚合了全局信息，靠一类 patch 即可完成任务
                                * 可以解释为什么打乱 image tokens 顺序后影响较小。因为位置信息最需要的地方是在 ViT 对孤立图像 patches 的处理过程中；当 ViT 处理完成，位置相对信息已经融入了每个 patch，所以后续就不再需要额外的位置信息了。 
                * 位置编码究竟是如何实现的？
                    * 理论推导
                    * 方向向量的概念
                    * 聚类检验
                    * 维度分组检验？
            * 模态交互
            * 应用价值
                * token compression
                * RoPE 存在的问题？
                    * attention 干预？
        * 再次审视，VLM 如何辨别方位
            * llava
                * 加性位置编码。位置编码已经很好编码了位置坐标(24*24)。问题是我们不知道在输入端加上这种编码后，这种位置信息是怎么参与推理的。对于
            * qwen
        * 待办
            * 测试 llava whatsup，换个问法
            * 计算累积注意力
            * 重新筛选样本，测 rope 维度分组
    * 7.6
        * rope 维度分组
            * 没准就是这么奇怪，纵坐标就是没有横坐标那么有分离度，可能是训练数据的原因
            * 可以再研究一下总的（不分组）注意力分数
                * 累积注意力：可通过通项公式计算。这种方法的出发点在于锁定某一物体初始的特征，观察在每一层中这个物体在另一个物体的表征中的比例的变化；
                * 不累积：把每一层的处理结果都看作一个全新的物体表征，层与层之间的注意力操作是相互独立的；
        * 模态交互
            * 空间感知研究的是空间信息的表征，模态交互研究的是物体信息和空间信息的利用
            * 物体识别： Is there a ...?
            * 空间感知
                * 一维绝对位置编码
                * 二维 rope
        * 发现了一个严重的问题：object patch ids 都是针对 merge 以后的图像序列，而不是 ViT 里的仅经过 patchify 以后的序列。好在之前的所有实验都是考虑的 after connector 的表征，没有被这个 bug 攻击。而在最近的 rope dim group 实验中，需要用到 merge 之前的 object patch ids，这就需要重新计算 merge 之前的物体位置。
    * 7.7
        * 昨天改了代码，分组（X, Y）的图变得合理了，现在要解决的问题是：
            * ！累积注意力。
                * 绘制了四个方向对应的 n in s 图，发现成分比例值区分得越开，两种方向辨别得就越好。
                    * !! qwen2_vl_2b: left 和 right 重叠了。所有正确率仅有 0.3718，而 behind/in front of 则达到了 0.7231
                    * !! qwen2_vl_7b: behind/in front of 重叠了。所有正确率仅有 0.7758，而 left/right 则达到了 0.9743
            * 这么分析的合理性？
        * 模态交互可以拆解成两个问题，一个是物体识别，对应的表征是 low level 的特征；另一个是空间感知，对应的是 high level（需要 patch 间通过 attention 交互，不过也不一定，通过单个 patch 的表征可能也能完成）的特征。
            * 物体识别，感觉可以直接看 attention，结合之前信息流动的相关研究进行分析
            * 空间感知，重在讨论位置关系的表征是怎么被利用的
            * rope
                * 是否可以观察 last token 的表征，看 umbed(left) - unembed(right) 在其中的分量（这一点验证出来肯定是完全符合预期的），以及方向向量在其中的分量（这个很重要，如果符号预期，即相似度越来越大，就可以作为方向向量存在性的证明）
                    * 方向向量是两个物体的表征相减，这一操作单从整个 attention 模块来看，是不存在的，因为 attention 分数是正的，并且作为加权系数指导各个 patch 的表征求和。所以要拆开来看，首先 attention heads 得到各自的输出，然后通过 W_O 来实现各个 attention head 信息的聚合。
            * 一维绝对位置编码
                * 
        * 实际应用？
            * 从空间感知入手：从 rope attention 解构的角度入手，提升空间推理能力？分别计算 X 和 Y 对应的 attention pattern，然后突出各自的特征，然后再合成为一个 pattern。这其实是一种特征缩放。
            * 从模态交互入手：？
    * 7.8/9/10
        * 昨天的实验是错的，因为输入 attention 的 hidden states 忘记加 norm 了
            * 改正后，发现 x, y 分开看的图变丑了，尤其是左和右的图，左右重叠在一起了
            * !! 想了一下，不能通过 attention 值的大小来判断方位。例如同样是左，一个离 nucleus 很近，另一个很远，attention 分数可能就不一样。要想辨别左右，肯定需要一个二维的统计量，例如正和负。此外，模型是否会根据特定的维度来辨别左右和上下？
                * 这也为方向向量的存在性提供了新的证据！
                * 方向向量是否可以是两个物体表征的加和？因为 attention 交互时，没有负的注意力分数。此外，LLM 阶段的图像位置编码已失效，因此模型理论上可以平权地关注到 satellite 和 nucleus 的表征。 
                * But, 在可视化的时候，还是要用相减（已验证）。因为相减能够突出两个方向向量的差异。因为能减去相同的背景。
        * 模态交互可以拆解成两个问题，一个是物体识别，对应的表征是 low level 的特征；另一个是空间感知，对应的是 high level（需要 patch 间通过 attention 交互，不过也不一定，通过单个 patch 的表征可能也能完成）的特征。
            * 物体识别，感觉可以直接看 attention，结合之前信息流动的相关研究进行分析
            * 空间感知，重在讨论位置关系的表征是怎么被利用的
            * rope
                * 是否可以观察 last token 的表征，看 umbed(left) - unembed(right) 在其中的分量（这一点验证出来肯定是完全符合预期的），以及方向向量在其中的分量（这个很重要，如果符号预期，即相似度越来越大，就可以作为方向向量存在性的证明）、
                    * 方向向量是两个物体的表征相减，这一操作单从整个 attention 模块来看，是不存在的，因为 attention 分数是正的，并且作为加权系数指导各个 patch 的表征求和。所以要拆开来看，首先 attention heads 得到各自的输出，然后通过 W_O 来实现各个 aattention head 信息的聚合。
            * 一维绝对位置编码
                * 
        * 实际应用？
            * 从空间感知入手：从 rope attention 解构的角度入手，提升空间推理能力？分别计算 X 和 Y 对应的 attention pattern，然后突出各自的特征，然后再合成为一个 pattern。这其实是一种特征缩放。
            * 从模态交互入手：？
        * ！！！物体识别的过程，整体可以归结为：心理学里的格式塔认知过程
            * [格式塔知觉组织原则](https://www.sohu.com/a/151032652_99921930)
        * 待办
            * ！读 MMLM knwows where to look，思考是否能结合 attention，对图像中的特定表征进行增强？最好能解耦横纵坐标对应的注意力
                * 可以先用 bounding box 试一试，后续再迁移到利用注意力代替 bbox
                * 该怎么对表征进行干预呢？发现决定方向之间差异的交叉项（qa kb 交叉项，即物体之间的相互注意力）的系数是 sin，sin 在第一象限通常比 cos 小，是否能从此处入手？
            * 能否证明左和上的方向向量各自独有的分量是正交的
                * 太麻烦，懒得验证
            * 测 qwen2_vl_2b 解耦 x,y 注意力的准确率
                * lr: 0.352, 0.336, 0.342
            * 前些天做的游程编码发现的解码成符号的 tokens，是否是 register tokens？
                * 检查 norm，在物体识别的分析过程中要提一嘴
                * 后面应用的那一部分也要讨论一下
                * 还可能是 icml 25 的研究（MLLM 极大值）
    * 7.11/12
        * 物体识别
            * 前些天做的游程编码发现的解码成符号的 tokens，是否是 register tokens？
                * 检查 norm，在物体识别的分析过程中要提一嘴
                * 后面应用的那一部分也要讨论一下
                * 还可能是 icml 25 的研究（MLLM 极大值）
            * 重新审视物体识别的 seg_with_activations
                * 物体识别是一个格式塔认知的过程。在浅层，模型先观察到最低层的特征，如颜色、质地等。从中层到高层，模型逐渐学会消歧，即将相近的物体组合起来理解，知道了某一 patch 所属的物体的具体名称；
                * 聚类图与分割图：聚类图一方面是用来呈现物体识别的早期阶段，即模型脑海中的表征仍然是可以通过欧氏距离来计算相似度的，而深层阶段由于每个 patch 都掺杂了复杂的高层语义信息，模型无法通过欧氏距离来线性地度量表征之间的相似度。所以这时需要一个强有力的工具，来呈现模型对高层语义的处理过程，那就是分割图。
                    * 从数学角度证明了归一化后欧氏距离和余弦相似度的等价性
                    * 这也说明，在深层阶段，仅仅靠注意力分数/相似度和欧氏距离是无法很好地度量表征相似度的，因为它们都是线性空间中的度量工具
        * 空间感知
            * 发现模型在 whatsup 上测试的 failure cases 主要是混淆共线方向的方位，比如左右混淆等，原因是两种方位对应的表征的区别在于交叉相乘再相减再与 sin 相乘的项。首先，交叉相乘再相减，就很小了；其次，当 i 也就是维度组数增大，频率(theta)值显著降低，导致 sin 值对距离变化不敏感。
                * training-free 的解决方案：让 gemini 帮着想，发现了一些缩放的方法，其目的在于在维度组数较低时，基本不做缩放，而在维度组数较高时，对低频进行成倍的缩放。
                    * 昨天晚上开启了网格搜索模式
                * training-based 的方法：缩放版 rope + SFT
                    * 用 SpaRE 训？还是之间用 GQA 训？
                        * 先用 GQA 训，SPARE 里面的 answer 就是英语里的完整回复，不是一个词敷衍了事，但仔细看仍有很多一个词的回答。
                    * 训练代码
                        * 7.12 晚完成（下午玩脱逃者2）。qwen2_vl_2b 全量调占 27952M，LoRA 占 14380(r=16)
                        * bug 集锦
                            * python -u，才能在使用 tee 的情况下，在终端打印训练信息
                            * 加载 ckpt 时，需要把 preprocessor_config.json 文件复制到 ckpt 目录下
        * 模态交互
            * 研究一下 llm 阶段物体识别的情况
                * 之前做了(llm_layer_logit_lens)
            * 老王说的基于语义的游程编码
            * 老王推荐的论文
                * [SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs](https://arxiv.org/pdf/2306.17842)
                * [Beyond Text: Frozen Large Language Models in Visual Signal Comprehension](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Beyond_Text_Frozen_Large_Language_Models_in_Visual_Signal_Comprehension_CVPR_2024_paper.pdf)
        * 待办
            * rope scaling
                * 测 coco spatial
                * 测第二种方法？
                * 训练
                    * poly
                        * 参数写死：99, 6
                    * sigmoid
                        * 第一次训是自适应地训，但训练结束后发现模型调的结果跟初始化结果差很多
                        * 第二次直接规定 99, 0.6, 40，但测出来惨不忍睹
            * 游程编码
                * 新实验
                    * 发现在 qwen2-vl 和 qwen2_5_vl_3b 上测 mme 都很拉
                        * perception 没什么事，但 cognition 出问题
                        * 尝试换成 mean pooling
                * 整理实验结果
                * 基于语义的游程编码
            * 物体识别
                * 消歧的过程，能否在 pope 或 CHAIR 上测试？
            * 读论文
                * COMFORT 那篇，空间感知的数据集
    * 7.14
        * 待办
            * 游程编码
                * 测 qwen2_vl_7b：pope
                * 测 GQA
                * InternVL_2_5
            * 图像感知
    * 7.15
        * 昨天的实验,7b, lora(r=16, alpha=32)
            * p=6, alpha=99
            * p=8, alpha=99
            * p=10, alpha=99
            * p=12, alpha=99
            * p=6, alpha=149
            * p=6, alpha=49
        * 昨天测得不理想，今天新增实验
            * p=8, alpha=49
            * p=10, alpha=49
            * p=10, alpha=99, epoch=2
            * normal, epoch=2
        * 规律
            * 减小 poly_alpha，对 vsr 好；
    * 7.16
        * 目前的实验做得差不多了，遗留的问题和实验：
            * 物体识别
                * 物体识别的应用还有哪些？
                * runlength 编码
                    * 测 GQA
                    * 更细致的分析
                        * failure case
                    * 在 internvl 上优化速度
            * 位置编码
                * 测 GQA spatial
                * 绝对位置编码的分析
            * 模态交互
                * 
    * 8.7
        * 老子回归！
        * 论文开始写了
        * 昨天顺利实现了基于蒸馏的 token compression
        * 训练
            * llava 一开始爆显存，但是通过 del 中间变量，不爆显存了
            * qwen 还是爆显存，bsz=8 的时候没事
                * llava 是按照 qwen 那样，最后把 logits 搞成 (bsz*image_len, vocab_size)，相当于 bsz=1 的超长序列
                * 开了梯度累积，可以跑了: 8*4=32
                    * 损失不下降
                        * 调 lr ↓
                        * 调 alpha ↑
                        * 调 eval steps ↑
    * 8.8
        * llava: hidden_size=4096, vocab_size=32064
        * qwen2.5: hidden_size=3584, vocab_size=152064
        