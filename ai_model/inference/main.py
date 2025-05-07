# 主推理模块

import os
import json
import argparse
from symptom_recognition import SymptomRecognizer, QuestionGenerator
from risk_assessment import RiskAssessor

class MedicalAI:
    """医疗AI主类，整合症状识别和风险评估功能"""
    
    def __init__(self, config_path="config.json"):
        """初始化医疗AI
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化症状识别器
        symptom_model_path = self.config.get("symptom_model_path")
        self.symptom_recognizer = SymptomRecognizer(model_path=symptom_model_path)
        
        # 初始化风险评估器
        risk_model_path = self.config.get("risk_model_path")
        self.risk_assessor = RiskAssessor(model_path=risk_model_path)
        
        # 初始化问题生成器
        self.question_generator = QuestionGenerator()
        
        print("医疗AI系统初始化完成")
    
    def _load_config(self, config_path):
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return {
                "symptom_model_path": None,
                "risk_model_path": None
            }
    
    def process_user_input(self, user_input, conversation_history=None):
        """处理用户输入
        
        Args:
            user_input: 用户输入文本
            conversation_history: 对话历史
            
        Returns:
            处理结果，包含识别的症状、风险评估、下一个问题等
        """
        if conversation_history is None:
            conversation_history = {
                "recognized_symptoms": [],
                "answered_questions": [],
                "patient_info": {}
            }
        
        # 识别症状
        new_symptoms = self.symptom_recognizer.recognize(user_input)
        
        # 更新已识别的症状（去重并保留置信度最高的）
        updated_symptoms = self._update_symptoms(
            conversation_history["recognized_symptoms"], 
            new_symptoms
        )
        
        # 提取患者信息（年龄、性别等）
        patient_info = self._extract_patient_info(user_input, conversation_history["patient_info"])
        
        # 评估风险
        risk_assessment = self.risk_assessor.assess(updated_symptoms, patient_info)
        
        # 生成下一个问题
        next_question = self.question_generator.generate_next_question(
            updated_symptoms, 
            conversation_history["answered_questions"]
        )
        
        # 更新对话历史
        conversation_history["recognized_symptoms"] = updated_symptoms
        conversation_history["patient_info"] = patient_info
        conversation_history["answered_questions"].append(next_question)
        
        # 返回处理结果
        return {
            "symptoms": updated_symptoms,
            "risk_assessment": risk_assessment,
            "next_question": next_question,
            "conversation_history": conversation_history
        }
    
    def _update_symptoms(self, existing_symptoms, new_symptoms):
        """更新症状列表，去重并保留置信度最高的
        
        Args:
            existing_symptoms: 已存在的症状列表
            new_symptoms: 新识别的症状列表
            
        Returns:
            更新后的症状列表
        """
        # 创建症状字典，以症状名称为键
        symptom_dict = {}
        
        # 添加已存在的症状
        for symptom in existing_symptoms:
            symptom_dict[symptom["name"]] = symptom
        
        # 添加或更新新症状
        for symptom in new_symptoms:
            name = symptom["name"]
            if name not in symptom_dict or symptom["confidence"] > symptom_dict[name]["confidence"]:
                symptom_dict[name] = symptom
        
        # 转换回列表
        return list(symptom_dict.values())
    
    def _extract_patient_info(self, text, existing_info):
        """从文本中提取患者信息
        
        Args:
            text: 用户输入文本
            existing_info: 已存在的患者信息
            
        Returns:
            更新后的患者信息
        """
        # 复制已存在的信息
        patient_info = existing_info.copy()
        
        # 提取年龄信息（简单实现，实际应使用更复杂的NLP技术）
        import re
        age_patterns = [
            r'(\d+)\s*岁',
            r'年龄\s*(\d+)',
            r'(\d+)\s*年龄'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text)
            if match:
                patient_info["age"] = match.group(1)
                break
        
        # 提取性别信息
        gender_keywords = {
            "男": ["男", "男性", "男子", "先生", "男孩", "爸爸", "父亲", "丈夫"],
            "女": ["女", "女性", "女子", "女士", "太太", "女孩", "妈妈", "母亲", "妻子"]
        }
        
        for gender, keywords in gender_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    patient_info["gender"] = gender
                    break
            if "gender" in patient_info:
                break
        
        # 提取基础疾病信息
        disease_keywords = [
            "高血压", "糖尿病", "心脏病", "冠心病", "哮喘", "慢性阻塞性肺疾病",
            "肝炎", "肾病", "癌症", "肿瘤", "风湿", "类风湿", "痛风", "甲状腺"
        ]
        
        medical_history = patient_info.get("medical_history", [])
        
        for disease in disease_keywords:
            if disease in text and disease not in medical_history:
                medical_history.append(disease)
        
        if medical_history:
            patient_info["medical_history"] = medical_history
        
        return patient_info


# 命令行接口
def main():
    parser = argparse.ArgumentParser(description="智医通 - 医疗AI问诊系统")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--interactive", action="store_true", help="启用交互模式")
    args = parser.parse_args()
    
    # 初始化医疗AI
    medical_ai = MedicalAI(config_path=args.config)
    
    if args.interactive:
        # 交互模式
        conversation_history = None
        print("欢迎使用智医通AI问诊系统！请描述您的症状，输入'退出'结束对话。\n")
        
        while True:
            user_input = input("您: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("\n感谢使用智医通AI问诊系统，祝您健康！")
                break
            
            # 处理用户输入
            result = medical_ai.process_user_input(user_input, conversation_history)
            conversation_history = result["conversation_history"]
            
            # 输出识别的症状
            if result["symptoms"]:
                print("\n识别到的症状:")
                for symptom in result["symptoms"]:
                    print(f"- {symptom['name']} ({symptom.get('severity', '中度')})"
                          f": {symptom.get('description', '')}")
            
            # 输出风险评估
            risk = result["risk_assessment"]
            print(f"\n风险评估: {risk['risk_level']}")
            print(f"建议: {risk['advice']}")
            
            # 输出下一个问题
            next_question = result["next_question"]
            print(f"\n智医通: {next_question['text']}")
            
            if next_question.get("options"):
                print("选项:")
                for option in next_question["options"]:
                    print(f"- {option}")
    else:
        # 非交互模式，可用于API服务
        print("智医通AI系统已初始化完成，可通过API调用")


if __name__ == "__main__":
    main()