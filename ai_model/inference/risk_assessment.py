# 风险评估模块

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

class RiskAssessor:
    """风险评估类，用于评估患者症状的风险等级"""
    
    def __init__(self, model_path=None):
        """初始化风险评估器
        
        Args:
            model_path: 模型路径，如果为None则使用规则匹配方式
        """
        # 加载风险评估规则库（作为备用）
        self.risk_rules = self._load_risk_rules()
        
        # 加载微调后的模型
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self.load_model(model_path)
        else:
            print("未指定模型路径，将使用规则匹配方式进行风险评估")
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
            print(f"风险评估模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print("将使用规则匹配方式进行风险评估")
            self.model_loaded = False
    
    def _load_risk_rules(self):
        """加载风险评估规则库"""
        # 实际项目中应该从数据库或文件中加载
        # 此处为演示，返回一个简单的字典
        return {
            "高风险症状": [
                {
                    "name": "胸痛",
                    "conditions": [
                        "突发", "剧烈", "压榨感", "伴有出汗", "伴有呼吸困难", "向左臂放射"
                    ],
                    "risk_level": "高风险",
                    "advice": "立即就医，可能是心肌梗死的症状"
                },
                {
                    "name": "呼吸困难",
                    "conditions": [
                        "突发", "严重", "伴有胸痛", "伴有发绀", "伴有意识改变"
                    ],
                    "risk_level": "高风险",
                    "advice": "立即就医，可能是肺栓塞或严重哮喘发作"
                },
                {
                    "name": "头痛",
                    "conditions": [
                        "突发", "剧烈", "伴有呕吐", "伴有意识改变", "伴有颈部僵硬", "伴有发热"
                    ],
                    "risk_level": "高风险",
                    "advice": "立即就医，可能是脑膜炎或蛛网膜下腔出血"
                },
                {
                    "name": "腹痛",
                    "conditions": [
                        "突发", "剧烈", "右下腹", "伴有发热", "伴有呕吐", "压痛"
                    ],
                    "risk_level": "高风险",
                    "advice": "立即就医，可能是阑尾炎"
                }
            ],
            "中风险症状": [
                {
                    "name": "发热",
                    "conditions": [
                        "持续", "高烧", "伴有咳嗽", "伴有呼吸困难", "伴有乏力"
                    ],
                    "risk_level": "中风险",
                    "advice": "建议24小时内就医，可能是肺炎"
                },
                {
                    "name": "腹泻",
                    "conditions": [
                        "持续", "严重", "伴有血便", "伴有发热", "伴有腹痛"
                    ],
                    "risk_level": "中风险",
                    "advice": "建议24小时内就医，可能是细菌性肠炎"
                }
            ],
            "低风险症状": [
                {
                    "name": "咳嗽",
                    "conditions": [
                        "轻微", "无发热", "无呼吸困难", "无胸痛"
                    ],
                    "risk_level": "低风险",
                    "advice": "可先观察，如症状加重再就医，可能是普通感冒"
                },
                {
                    "name": "头痛",
                    "conditions": [
                        "轻微", "间歇性", "无发热", "无呕吐", "无视力改变"
                    ],
                    "risk_level": "低风险",
                    "advice": "可先休息和补充水分，如症状持续再就医，可能是紧张性头痛"
                }
            ]
        }
    
    def assess(self, symptoms, patient_info=None):
        """评估症状风险
        
        Args:
            symptoms: 症状列表，每个症状包含名称、严重程度等信息
            patient_info: 患者基本信息，如年龄、性别、基础疾病等
            
        Returns:
            风险评估结果，包含风险等级、建议等信息
        """
        # 如果模型已加载，使用模型进行推理
        if self.model_loaded and self.model and self.tokenizer:
            return self._assess_with_model(symptoms, patient_info)
        else:
            # 否则使用规则匹配
            return self._assess_with_rules(symptoms, patient_info)
    
    def _assess_with_model(self, symptoms, patient_info):
        """使用微调后的模型评估风险
        
        Args:
            symptoms: 症状列表
            patient_info: 患者基本信息
            
        Returns:
            风险评估结果
        """
        # 构建提示模板
        symptoms_text = ""
        for symptom in symptoms:
            severity = symptom.get("severity", "中度")
            symptoms_text += f"{severity}{symptom['name']}，"
        
        patient_text = ""
        if patient_info:
            age = patient_info.get("age", "未知")
            gender = patient_info.get("gender", "未知")
            medical_history = patient_info.get("medical_history", [])
            
            patient_text = f"患者信息：{age}岁，{gender}，"
            if medical_history:
                patient_text += f"有{','.join(medical_history)}等基础疾病，"
        
        prompt = f"<s>[INST] 患者症状：{symptoms_text}
{patient_text}
请评估风险等级并给出建议。 [/INST]"
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # 从模型输出中提取风险评估信息
        assessment_result = self._extract_assessment_from_response(response, symptoms)
        
        return assessment_result
    
    def _extract_assessment_from_response(self, response, symptoms):
        """从模型响应中提取风险评估信息
        
        Args:
            response: 模型生成的响应文本
            symptoms: 原始症状列表
            
        Returns:
            提取的风险评估结果
        """
        # 提取风险等级和建议
        risk_levels = ["低风险", "中风险", "高风险"]
        extracted_risk_level = "未知风险"
        extracted_advice = ""
        
        # 尝试解析JSON格式响应
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                assessment_data = json.loads(json_str)
                
                # 处理结构化的评估数据
                if isinstance(assessment_data, dict):
                    if "risk_level" in assessment_data:
                        extracted_risk_level = assessment_data["risk_level"]
                    if "advice" in assessment_data:
                        extracted_advice = assessment_data["advice"]
                    
                    return {
                        "risk_level": extracted_risk_level,
                        "advice": extracted_advice,
                        "details": assessment_data.get("details", ""),
                        "source": "model_structured"
                    }
        except Exception as e:
            print(f"解析JSON响应失败: {str(e)}，使用关键词匹配")
        
        # 如果JSON解析失败，使用关键词匹配
        for level in risk_levels:
            if level in response:
                extracted_risk_level = level
                break
        
        # 提取建议（简单实现，实际应使用更复杂的NLP技术）
        advice_keywords = ["建议", "应该", "需要", "可以"]
        for keyword in advice_keywords:
            if keyword in response:
                keyword_index = response.find(keyword)
                if keyword_index != -1:
                    # 提取包含建议关键词的句子
                    start_index = response.rfind("。", 0, keyword_index) + 1
                    if start_index == 0:
                        start_index = response.rfind("\n", 0, keyword_index) + 1
                    if start_index == 0:
                        start_index = 0
                    
                    end_index = response.find("。", keyword_index)
                    if end_index == -1:
                        end_index = len(response)
                    
                    extracted_advice = response[start_index:end_index].strip()
                    break
        
        # 如果没有提取到建议，使用整个响应作为建议
        if not extracted_advice:
            extracted_advice = response
        
        return {
            "risk_level": extracted_risk_level,
            "advice": extracted_advice,
            "source": "model_keyword"
        }
    
    def _assess_with_rules(self, symptoms, patient_info):
        """使用规则匹配评估风险
        
        Args:
            symptoms: 症状列表
            patient_info: 患者基本信息
            
        Returns:
            风险评估结果
        """
        # 默认为低风险
        risk_level = "低风险"
        advice = "症状较轻，可以先观察，如有不适请及时就医。"
        matched_rule = None
        
        # 提取症状名称和严重程度
        symptom_names = [s["name"] for s in symptoms]
        symptom_severities = [s.get("severity", "中度") for s in symptoms]
        
        # 检查是否匹配高风险规则
        for rule in self.risk_rules["高风险症状"]:
            if rule["name"] in symptom_names:
                # 获取症状在列表中的索引
                symptom_index = symptom_names.index(rule["name"])
                severity = symptom_severities[symptom_index]
                
                # 检查条件是否满足
                condition_matched = False
                for condition in rule["conditions"]:
                    # 检查严重程度
                    if condition in ["严重", "剧烈"] and severity in ["严重", "剧烈"]:
                        condition_matched = True
                        break
                    
                    # 检查其他症状是否同时存在
                    if condition.startswith("伴有"):
                        related_symptom = condition[2:]  # 去掉"伴有"前缀
                        for s in symptom_names:
                            if related_symptom in s:
                                condition_matched = True
                                break
                
                if condition_matched:
                    risk_level = "高风险"
                    advice = rule["advice"]
                    matched_rule = rule
                    break
        
        # 如果没有匹配高风险规则，检查中风险规则
        if risk_level != "高风险":
            for rule in self.risk_rules["中风险症状"]:
                if rule["name"] in symptom_names:
                    # 获取症状在列表中的索引
                    symptom_index = symptom_names.index(rule["name"])
                    severity = symptom_severities[symptom_index]
                    
                    # 检查条件是否满足
                    condition_matched = False
                    for condition in rule["conditions"]:
                        # 检查严重程度
                        if condition in ["中度", "持续"] and severity in ["中度", "持续"]:
                            condition_matched = True
                            break
                        
                        # 检查其他症状是否同时存在
                        if condition.startswith("伴有"):
                            related_symptom = condition[2:]  # 去掉"伴有"前缀
                            for s in symptom_names:
                                if related_symptom in s:
                                    condition_matched = True
                                    break
                    
                    if condition_matched:
                        risk_level = "中风险"
                        advice = rule["advice"]
                        matched_rule = rule
                        break
        
        # 考虑患者基本信息对风险的影响
        if patient_info:
            age = patient_info.get("age")
            if age and (int(age) < 12 or int(age) > 65):
                # 儿童和老年人风险等级提高
                if risk_level == "低风险":
                    risk_level = "中风险"
                    advice = "考虑到您的年龄因素，建议尽快就医进行检查。"
                
            # 考虑基础疾病
            medical_history = patient_info.get("medical_history", [])
            high_risk_conditions = ["高血压", "糖尿病", "冠心病", "慢性阻塞性肺疾病", "哮喘", "免疫抑制"]
            
            for condition in high_risk_conditions:
                if any(condition in history for history in medical_history):
                    # 有高风险基础疾病，风险等级提高
                    if risk_level == "低风险":
                        risk_level = "中风险"
                        advice = "考虑到您的基础疾病，建议尽快就医进行检查。"
                    elif risk_level == "中风险":
                        risk_level = "高风险"
                        advice = "考虑到您的基础疾病，建议立即就医。"
                    break
        
        return {
            "risk_level": risk_level,
            "advice": advice,
            "matched_rule": matched_rule,
            "source": "rules"
        }


# 测试代码
if __name__ == "__main__":
    # 初始化风险评估器
    assessor = RiskAssessor()
    
    # 测试症状
    test_symptoms = [
        {
            "name": "头痛",
            "confidence": 0.9,
            "severity": "严重"
        },
        {
            "name": "发热",
            "confidence": 0.8,
            "severity": "中度"
        }
    ]
    
    # 测试患者信息
    test_patient_info = {
        "age": 70,
        "gender": "男",
        "medical_history": ["高血压", "糖尿病"]
    }
    
    # 评估风险
    assessment = assessor.assess(test_symptoms, test_patient_info)
    
    # 输出结果
    print(f"风险等级: {assessment['risk_level']}")
    print(f"建议: {assessment['advice']}")
