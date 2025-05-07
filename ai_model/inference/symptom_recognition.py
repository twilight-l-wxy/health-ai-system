# 症状识别模块

import os
import json
import numpy as np
import torch
import jieba
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

class SymptomRecognizer:
    """症状识别类，用于从用户描述中识别症状关键词"""
    
    def __init__(self, model_path=None):
        """初始化症状识别器
        
        Args:
            model_path: 模型路径，如果为None则使用配置中的默认路径
        """
        # 加载症状关键词和同义词库（作为备用）
        self.symptom_keywords = self._load_symptom_keywords()
        self.symptom_synonyms = self._load_symptom_synonyms()
        
        # 加载微调后的模型
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self.load_model(model_path)
        else:
            print("未指定模型路径，将使用规则匹配方式进行症状识别")
            self.model_loaded = False
    
    def load_model(self, model_path):
        """加载模型
        
        Args:
            model_path: 模型路径
        """
        try:
            print(f"正在加载模型: {model_path}")
            
            # 检查是否为ONNX模型
            if os.path.exists(os.path.join(model_path, "model.onnx")):
                # 加载ONNX模型
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = ORTModelForCausalLM.from_pretrained(model_path)
                print("已加载ONNX优化模型")
            else:
                # 加载普通模型
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                print("已加载标准模型")
            
            self.model_loaded = True
            print(f"症状识别模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print("将使用规则匹配方式进行症状识别")
            self.model_loaded = False
    
    def _load_symptom_keywords(self):
        """加载症状关键词库"""
        # 实际项目中应该从数据库或文件中加载
        # 此处为演示，返回一个简单的字典
        return {
            "头痛": {
                "body_part": "头部",
                "severity_levels": ["轻微", "中度", "严重"],
                "common_causes": ["疲劳", "压力", "偏头痛", "感冒", "鼻窦炎"]
            },
            "发热": {
                "body_part": "全身",
                "severity_levels": ["低烧", "中烧", "高烧"],
                "common_causes": ["感染", "炎症", "感冒", "流感"]
            },
            "咳嗽": {
                "body_part": "呼吸道",
                "severity_levels": ["轻微", "中度", "严重"],
                "common_causes": ["感冒", "过敏", "哮喘", "支气管炎"]
            },
            "腹痛": {
                "body_part": "腹部",
                "severity_levels": ["轻微", "中度", "严重"],
                "common_causes": ["消化不良", "胃炎", "肠炎", "阑尾炎"]
            },
            "恶心": {
                "body_part": "消化系统",
                "severity_levels": ["轻微", "中度", "严重"],
                "common_causes": ["消化不良", "晕动病", "食物中毒", "胃炎"]
            }
        }
    
    def _load_symptom_synonyms(self):
        """加载症状同义词库"""
        # 实际项目中应该从数据库或文件中加载
        # 此处为演示，返回一个简单的字典
        return {
            "头痛": ["头疼", "脑袋痛", "头部疼痛", "颅痛", "偏头痛", "头晕", "头胀"],
            "发热": ["发烧", "体温升高", "发高烧", "低烧", "中烧", "高烧", "发热感", "体温异常"],
            "咳嗽": ["咳", "干咳", "湿咳", "久咳", "咳痰", "咳个不停", "咳嗽不止", "刺激性咳嗽", "痰多"],
            "腹痛": ["肚子痛", "肚子疼", "腹部疼痛", "肚痛", "胃痛", "肠痛", "腹部不适", "腹胀", "腹部绞痛"],
            "恶心": ["想吐", "反胃", "胃部不适", "恶心感", "作呕", "呕吐感", "胃里翻腾"],
            "头晕": ["眩晕", "晕眩", "天旋地转", "站立不稳", "晕厥感"],
            "乏力": ["疲劳", "无力", "虚弱", "精神不振", "浑身无力", "没精神"],
            "胸闷": ["胸部压迫感", "呼吸不畅", "气短", "胸口闷", "胸口紧"],
            "心悸": ["心跳加速", "心慌", "心跳不规律", "心脏怦怦跳"],
            "腹泻": ["拉肚子", "肠胃炎", "水样便", "大便稀", "肠炎", "急性腹泻"],
            "便秘": ["排便困难", "大便干结", "排便不畅", "大便硬"],
            "关节痛": ["关节疼痛", "骨头痛", "关节肿痛", "关节不适", "骨痛"]
        }
    
    def recognize(self, text):
        """从文本中识别症状
        
        Args:
            text: 用户描述的症状文本
            
        Returns:
            识别出的症状列表，每个症状包含名称、置信度等信息
        """
        # 如果模型已加载，使用模型进行推理
        if self.model_loaded and self.model and self.tokenizer:
            return self._recognize_with_model(text)
        else:
            # 否则使用规则匹配
            return self._recognize_with_rules(text)
    
    def _recognize_with_model(self, text):
        """使用微调后的模型识别症状
        
        Args:
            text: 用户描述的症状文本
            
        Returns:
            识别出的症状列表
        """
        # 构建提示模板
        prompt = f"<s>[INST] 患者描述: {text} [/INST]"
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # 从模型输出中提取症状信息
        recognized_symptoms = self._extract_symptoms_from_response(response, text)
        
        return recognized_symptoms
    
    def _extract_symptoms_from_response(self, response, original_text):
        """从模型响应中提取症状信息
        
        Args:
            response: 模型生成的响应文本
            original_text: 原始输入文本
            
        Returns:
            提取的症状列表
        """
        # 提取症状关键词
        recognized_symptoms = []
        
        # 尝试解析JSON格式响应
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                symptoms_data = json.loads(json_str)
                
                # 处理结构化的症状数据
                if isinstance(symptoms_data, dict) and "symptoms" in symptoms_data:
                    # 如果是标准格式的JSON响应
                    for symptom in symptoms_data["symptoms"]:
                        symptom_name = symptom.get("name")
                        if symptom_name in self.symptom_keywords:
                            recognized_symptoms.append({
                                "name": symptom_name,
                                "confidence": 0.95,  # 结构化数据置信度更高
                                "severity": symptom.get("severity", "中度"),
                                "info": self.symptom_keywords[symptom_name],
                                "possible_causes": symptom.get("possible_causes", []),
                                "source": "model_structured"
                            })
                elif isinstance(symptoms_data, list):
                    # 如果是症状列表
                    for symptom in symptoms_data:
                        if isinstance(symptom, dict) and "name" in symptom:
                            symptom_name = symptom["name"]
                            if symptom_name in self.symptom_keywords:
                                recognized_symptoms.append({
                                    "name": symptom_name,
                                    "confidence": 0.9,
                                    "severity": symptom.get("severity", "中度"),
                                    "info": self.symptom_keywords[symptom_name],
                                    "possible_causes": symptom.get("possible_causes", []),
                                    "source": "model_structured"
                                })
        except Exception as e:
            print(f"解析JSON响应失败: {str(e)}，使用关键词匹配")
        
        # 如果JSON解析失败或没有识别出症状，使用关键词匹配
        if not recognized_symptoms:
            # 尝试从响应中提取症状
            for symptom in self.symptom_keywords.keys():
                if symptom in response:
                    confidence = 0.9  # 模型直接提到的症状，置信度高
                    recognized_symptoms.append({
                        "name": symptom,
                        "confidence": confidence,
                        "severity": self._extract_severity(response, symptom),
                        "info": self.symptom_keywords[symptom],
                        "source": "model_keyword"
                    })
        
        # 如果模型没有识别出症状，尝试使用规则匹配
        if not recognized_symptoms:
            rule_symptoms = self._recognize_with_rules(original_text)
            for symptom in rule_symptoms:
                symptom["confidence"] *= 0.8  # 规则匹配的置信度稍低
                symptom["source"] = "rules"
                recognized_symptoms.append(symptom)
        
        # 添加症状描述
        for symptom in recognized_symptoms:
            symptom["description"] = self._generate_symptom_description(symptom["name"], symptom.get("severity", "中度"))
            
        return recognized_symptoms
        
    def _extract_severity(self, text, symptom):
        """从文本中提取症状严重程度
        
        Args:
            text: 响应文本
            symptom: 症状名称
            
        Returns:
            严重程度描述
        """
        severity_keywords = {
            "轻微": ["轻微", "轻度", "不严重", "略微", "稍微"],
            "中度": ["中度", "中等", "一般", "适中"],
            "严重": ["严重", "剧烈", "强烈", "重度", "难以忍受"]
        }
        
        # 查找症状附近的严重程度描述
        symptom_index = text.find(symptom)
        if symptom_index != -1:
            context = text[max(0, symptom_index - 20):min(len(text), symptom_index + 30)]
            
            for severity, keywords in severity_keywords.items():
                for keyword in keywords:
                    if keyword in context:
                        return severity
        
        return "中度"  # 默认为中度
    
    def _generate_symptom_description(self, symptom_name, severity):
        """生成症状描述
        
        Args:
            symptom_name: 症状名称
            severity: 严重程度
            
        Returns:
            症状描述文本
        """
        if symptom_name not in self.symptom_keywords:
            return f"{severity}{symptom_name}"
            
        info = self.symptom_keywords[symptom_name]
        body_part = info.get("body_part", "")
        common_causes = info.get("common_causes", [])
        
        description = f"{severity}{symptom_name}，位于{body_part}"
        
        if common_causes and len(common_causes) > 0:
            causes = "、".join(common_causes[:3])
            description += f"，常见原因包括{causes}等"
            
        return description
    
    def _recognize_with_rules(self, text):
        """使用规则匹配识别症状
        
        Args:
            text: 用户描述的症状文本
            
        Returns:
            识别出的症状列表
        """
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 识别症状关键词
        recognized_symptoms = []
        for word in words:
            # 直接匹配关键词
            if word in self.symptom_keywords:
                recognized_symptoms.append({
                    "name": word,
                    "confidence": 0.9,
                    "info": self.symptom_keywords[word]
                })
                continue
            
            # 匹配同义词
            for symptom, synonyms in self.symptom_synonyms.items():
                if word in synonyms:
                    recognized_symptoms.append({
                        "name": symptom,
                        "confidence": 0.8,
                        "info": self.symptom_keywords[symptom]
                    })
                    break
        
        # 去重（可能有同一症状的不同表达）
        unique_symptoms = {}
        for symptom in recognized_symptoms:
            name = symptom["name"]
            if name not in unique_symptoms or symptom["confidence"] > unique_symptoms[name]["confidence"]:
                unique_symptoms[name] = symptom
        
        return list(unique_symptoms.values())


class QuestionGenerator:
    """问题生成类，根据已知症状生成下一个问诊问题"""
    
    def __init__(self):
        """初始化问题生成器"""
        # 加载问题模板
        self.question_templates = self._load_question_templates()
        print("问题生成器初始化完成")
    
    def _load_question_templates(self):
        """加载问题模板"""
        # 实际项目中应该从数据库或文件中加载
        # 此处为演示，返回一个简单的字典
        return {
            "duration": "您的{symptom}持续了多长时间？",
            "severity": "您的{symptom}程度如何？",
            "frequency": "您的{symptom}多久发作一次？",
            "trigger": "有什么因素会加重或缓解您的{symptom}？",
            "associated": "除了{symptom}，您还有其他不适吗？"
        }
    
    def generate_next_question(self, recognized_symptoms, answered_questions):
        """生成下一个问诊问题
        
        Args:
            recognized_symptoms: 已识别的症状列表
            answered_questions: 已回答的问题列表
            
        Returns:
            下一个问题的字典，包含问题文本、类型等信息
        """
        if not recognized_symptoms:
            return {
                "text": "请描述您的症状",
                "type": "open"
            }
        
        # 选择主要症状（置信度最高的）
        main_symptom = max(recognized_symptoms, key=lambda x: x["confidence"])
        
        # 根据已回答的问题选择下一个问题类型
        answered_types = [q["type"] for q in answered_questions]
        
        if "duration" not in answered_types:
            question_type = "duration"
        elif "severity" not in answered_types:
            question_type = "severity"
        elif "frequency" not in answered_types:
            question_type = "frequency"
        elif "trigger" not in answered_types:
            question_type = "trigger"
        else:
            question_type = "associated"
        
        # 生成问题文本
        question_text = self.question_templates[question_type].format(symptom=main_symptom["name"])
        
        # 根据问题类型设置选项
        options = None
        if question_type == "duration":
            options = ["不到24小时", "1-3天", "4-7天", "一周以上", "一个月以上"]
        elif question_type == "severity":
            options = ["轻微", "中度", "严重"]
        elif question_type == "frequency":
            options = ["持续存在", "每天多次", "每天一次", "每周几次", "偶尔发作"]
        
        return {
            "text": question_text,
            "type": question_type,
            "options": options
        }


# 测试代码
if __name__ == "__main__":
    # 初始化症状识别器
    recognizer = SymptomRecognizer()
    
    # 测试症状识别
    test_text = "我最近总是头疼，而且有时候会发烧，感觉很不舒服"
    symptoms = recognizer.recognize(test_text)
    print("识别出的症状:")
    for symptom in symptoms:
        print(f"- {symptom['name']} (置信度: {symptom['confidence']})")
    
    # 初始化问题生成器
    generator = QuestionGenerator()
    
    # 测试问题生成
    next_question = generator.generate_next_question(symptoms, [])
    print(f"\n下一个问题: {next_question['text']}")
    if next_question['options']:
        print("选项:")
        for option in next_question['options']:
            print(f"- {option}")