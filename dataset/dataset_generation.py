from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import random
from typing import List
from models import TraumaAssessmentDataset, ConversationTurn, AssessmentData
import os
from dotenv import load_dotenv

load_dotenv()

class TraumaDatasetGenerator:
    def __init__(self, model_name="openai/gpt-4o-mini", temperature=0.7):
        self.model = ChatOpenAI(model=model_name, 
                                temperature=temperature,
                                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                                openai_api_base=os.getenv("OPENROUTER_BASE_URL"),)
        self.structured_model = self.model.with_structured_output(TraumaAssessmentDataset)
        
    def generate_single_case(self, scenario_context: str) -> TraumaAssessmentDataset:
        """Generate a single trauma assessment case"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", f"Generate a realistic trauma assessment case based on this scenario: {scenario_context}")
        ])
        
        chain = prompt | self.structured_model
        return chain.invoke({"scenario": scenario_context})
    
    def generate_batch(self, num_cases: int = 10, batch_size: int = 40) -> List[TraumaAssessmentDataset]:
        """Generate multiple cases for fine-tuning dataset
        
        Args:
            num_cases: Total number of cases to generate
            batch_size: Number of cases to process in each batch
        """
        scenarios = self._get_scenario_templates()
        selected_scenarios = [random.choice(scenarios) for _ in range(num_cases)]
        generated_cases = []
        
        # Create prompt chain once
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Generate a realistic trauma assessment case based on this scenario: {scenario}")
        ])
        chain = prompt | self.structured_model
        
        # Process in batches
        for i in range(0, num_cases, batch_size):
            batch_scenarios = selected_scenarios[i:i + batch_size]
            try:
                # Batch process current chunk of scenarios
                inputs = [{"scenario": scenario} for scenario in batch_scenarios]
                batch_results = chain.batch(inputs)
                generated_cases.extend(batch_results)
                print(f"Generated cases {i+1}-{min(i+batch_size, num_cases)} of {num_cases}")
            except Exception as e:
                print(f"Error during batch generation (cases {i+1}-{min(i+batch_size, num_cases)}): {e}")
                continue
        
        print(f"Successfully generated {len(generated_cases)} out of {num_cases} requested cases")
        return generated_cases
    
    def save_dataset(self, cases: List[TraumaAssessmentDataset], filename: str):
        """Save dataset in format suitable for fine-tuning"""
        
        training_data = []
        
        for case in cases:
            # Convert to fine-tuning format
            training_example = {
                "messages": [
                    {"role": turn.role, "content": turn.content} 
                    for turn in case.conversation
                ]
            }
            training_data.append(training_example)
            
            # Also create report generation examples
            conversation_text = self._format_conversation_for_report(case.conversation)
            report_example = {
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Based on this conversation, generate a professional trauma assessment report:\n\n{conversation_text}"
                    },
                    {
                        "role": "assistant",
                        "content": self._format_assessment_output(case.assessment_data)
                    }
                ]
            }
            training_data.append(report_example)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(training_data)} training examples to {filename}")
    
    def _get_system_prompt(self) -> str:
        return """You are an expert in child trauma psychology and assessment. Your task is to generate realistic, culturally sensitive conversations between parents/caregivers and an AI assistant, along with corresponding professional assessment reports.

LANGUAGE REQUIREMENTS:
- When parents speak in their local language/dialect, respond in the same language/dialect
- Support conversations in:
  * Arabic (Palestinian/Levantine dialect) for Palestinian/Syrian families
  * Ukrainian language for Ukrainian families
  * Arabic (Sudanese dialect) for Sudanese families
  * English for any family comfortable with it
- Use authentic cultural expressions and idioms appropriate to each dialect
- Mirror the language style and formality level used by the parent

CONVERSATION REQUIREMENTS:
- Create natural, empathetic conversations where the AI asks appropriate follow-up questions
- AI should gather information about: behavioral changes, sleep patterns, emotional responses, social interactions, academic performance, physical symptoms
- Parents should provide realistic, detailed observations about their child's trauma responses
- Include cultural context (Gaza/Palestine, Ukraine, Sudan, Syria, etc.)
- Conversations should be 8-15 exchanges long
- AI should be supportive, non-judgmental, and professionally appropriate
- Include clarifying questions about specific incidents, timeframes, and severity
- Reflect authentic local ways of expressing concerns and emotions

ASSESSMENT REPORT REQUIREMENTS:
- All reports should be written in clear, professional English
- Parent observations: Professional English summary of what parents reported
- AI analysis: Professional interpretation of trauma indicators, behavioral patterns, and psychological impact
- Severity score: 1-3 (mild), 4-6 (moderate), 7-8 (severe), 9-10 (critical)
- Risk indicators: Specific flags like "sleep_disturbances", "aggressive_behavior", "withdrawal", "regression", "hypervigilance", "dissociation", "self_harm_risk", etc.
- Cultural context: How trauma manifests in specific cultural settings, relevant cultural healing practices, family dynamics

SCENARIO VARIATIONS:
- Child age: 4-17 years old
- Trauma types: Not limited to but including - bombing/airstrikes, displacement, family separation, witnessing violence, loss of home/school, ongoing occupation stress, checkpoint trauma, etc.
- Severity levels: from mild adjustment issues to severe PTSD symptoms
- Cultural backgrounds: Not limited to but including Palestinian, Ukrainian, Sudanese, Syrian families
- Family structures: single parent, extended family, refugee situations, traditional family units
- Socioeconomic situations: urban/rural settings, refugee camps, temporary housing, established communities
- Educational contexts: formal schooling, interrupted education, alternative education settings

IMPORTANT: Make conversations realistic and culturally authentic - parents may be hesitant to share, might downplay symptoms, could be dealing with their own trauma. AI should be patient and understanding. Use appropriate cultural references when discussing mental health and trauma."""

    def _get_scenario_templates(self) -> List[str]:
        """Generate dynamic scenarios instead of using fixed templates"""
        # Core components for scenario generation
        backgrounds = [
            {"culture": "Palestinian", "locations": ["Gaza", "West Bank", "Jerusalem", "Refugee camp in Lebanon"],
             "dialects": ["Palestinian Arabic", "Levantine Arabic"]},
            {"culture": "Ukrainian", "locations": ["Kyiv", "Mariupol", "Kharkiv", "Refugee centers in Poland/Germany"],
             "dialects": ["Ukrainian"]},
            {"culture": "Sudanese", "locations": ["Khartoum", "Displacement camps", "Rural villages"],
             "dialects": ["Sudanese Arabic"]},
            {"culture": "Syrian", "locations": ["Aleppo", "Damascus", "Refugee camps in Lebanon/Turkey"],
             "dialects": ["Syrian Arabic", "Levantine Arabic"]}
        ]
        
        age_groups = [
            {"range": "4-6", "context": "early childhood"},
            {"range": "7-9", "context": "middle childhood"},
            {"range": "10-12", "context": "pre-teen"},
            {"range": "13-15", "context": "early teen"},
            {"range": "16-17", "context": "late teen"}
        ]
        
        family_structures = [
            "single mother caring for children",
            "single father with extended family support",
            "grandparents as primary caregivers",
            "traditional nuclear family under stress",
            "separated family due to conflict",
            "living with extended family in shared housing",
            "unaccompanied minor with caretaker",
            "blended family in temporary housing"
        ]
        
        trauma_contexts = [
            "recent bombing/airstrike exposure",
            "chronic checkpoint-related trauma",
            "displacement and loss of home",
            "separation from family members",
            "witnessing violence in community",
            "school destruction or closure",
            "loss of friends or relatives",
            "living under occupation stress",
            "economic hardship and instability",
            "interrupted education",
            "multiple relocations",
            "arrest of family member",
            "home demolition",
            "ongoing military presence",
            "refugee camp conditions"
        ]
        
        # Generate random scenarios
        scenarios = []
        for _ in range(15):  # Generate 15 base scenarios
            background = random.choice(backgrounds)
            age = random.choice(age_groups)
            structure = random.choice(family_structures)
            # Pick 1-3 trauma contexts
            contexts = random.sample(trauma_contexts, random.randint(1, 3))
            
            # Construct scenario with cultural and linguistic context
            scenario = f"{age['range']}-year-old {background['culture']} child from {random.choice(background['locations'])}, "
            scenario += f"speaking {random.choice(background['dialects'])}. "
            scenario += f"Family situation: {structure}. "
            scenario += f"Context: {'; '.join(contexts)}."
            
            scenarios.append(scenario)
            
        return scenarios
    
    def _format_conversation_for_report(self, conversation: List[ConversationTurn]) -> str:
        """Format conversation for report generation training"""
        formatted = []
        for turn in conversation:
            if turn.role == "user":
                formatted.append(f"Parent: {turn.content}")
            else:
                formatted.append(f"AI Assistant: {turn.content}")
        return "\n\n".join(formatted)
    
    def _format_assessment_output(self, assessment: AssessmentData) -> str:
        """Format assessment data for training"""
        risk_indicators_str = ", ".join(assessment.risk_indicators)
        return f"""TRAUMA ASSESSMENT REPORT

PARENT OBSERVATIONS:
{assessment.parent_observations}

AI ANALYSIS:
{assessment.ai_analysis}

SEVERITY SCORE: {assessment.severity_score}/10

RISK INDICATORS: {risk_indicators_str}

CULTURAL CONTEXT:
{assessment.cultural_context}

END OF REPORT"""

# Usage Example
def main():
    generator = TraumaDatasetGenerator(model_name="openai/gpt-4o-mini")
    
    # Generate training dataset
    print("Generating trauma assessment dataset...")
    cases = generator.generate_batch(num_cases=1000)  # Generate 50 cases
    
    # Save for fine-tuning
    generator.save_dataset(cases, "trauma_assessment_training.jsonl")
    
    # Also save raw structured data for analysis
    with open("trauma_cases_structured.json", "w", encoding="utf-8") as f:
        cases_dict = [case.dict() for case in cases]
        json.dump(cases_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(cases)} trauma assessment cases")
    print("Files saved: trauma_assessment_training.jsonl, trauma_cases_structured.json")

if __name__ == "__main__":
    main()