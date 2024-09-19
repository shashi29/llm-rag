import json
import os
import logging
import zlib
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from config import Settings
from typing import List, Dict
from diskcache import Cache, Disk, UNKNOWN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillMatchBreakdown(BaseModel):
    Technical_Skills: str = Field(description="Percentage match and list of top matches and gaps for technical skills")
    Soft_Skills: str = Field(description="Percentage match and list of top matches and gaps for soft skills")
    Certifications: str = Field(description="Percentage match and list of top matches and gaps for certifications")

class ExperienceRelevance(BaseModel):
    Years_of_Relevant_Experience: str = Field(description="Years of relevant experience out of required")
    Experience_Quality: str = Field(description="Quality of experience (High/Medium/Low)")

class ProjectAlignment(BaseModel):
    Project_Name: str = Field(description="Name of the relevant project")
    Description: str = Field(description="Brief description of the project")
    Relevance_Score: float = Field(description="Relevance score of the project")

class ResumeJobMatch(BaseModel):
    Overall_Match_Score: str = Field(description="Overall match score as a percentage")
    Skill_Match_Breakdown: SkillMatchBreakdown = Field(description="Detailed breakdown of skill matches")
    Experience_Relevance: ExperienceRelevance = Field(description="Analysis of experience relevance")
    Project_Alignment: List[ProjectAlignment] = Field(description="List of relevant projects and their alignment")
    Key_Strengths: List[str] = Field(description="List of key strengths of the candidate")
    Notable_Gaps: List[str] = Field(description="List of notable gaps in the candidate's profile")
    Overall_Assessment: str = Field(description="Qualitative summary of the candidate's fit")
    Recommendations: Dict[str, str] = Field(description="Recommendations for next steps")

class JSONDisk(Disk):
    def __init__(self, directory, compress_level=1, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        json_bytes = json.dumps(key).encode('utf-8')
        data = zlib.compress(json_bytes, self.compress_level)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return json.loads(zlib.decompress(data).decode('utf-8'))

    def store(self, value, read, key=UNKNOWN):
        if not read:
            json_bytes = json.dumps(value).encode('utf-8')
            value = zlib.compress(json_bytes, self.compress_level)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = json.loads(zlib.decompress(data).decode('utf-8'))
        return data

class ResumeJobMatchingService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=Settings.OPENAI_API_KEY,
                                temperature=Settings.OPENAI_TEMPERATURE,
                                model_name=Settings.OPENAI_MODEL,
                                top_p=Settings.OPENAI_TOP_P)
        self.parser = JsonOutputParser(pydantic_object=ResumeJobMatch)
        
        self.prompt = PromptTemplate(
            template="""Analyze and quantify the relevance between the provided job description and candidate's resume to assess the candidate's suitability for the role.

            {format_instructions}

            Job Description: {job_description}
            Resume: {candidate_resume}

            Follow the analysis process outlined below:
            1. Skill Extraction and Matching (40% of total score)
            2. Experience Analysis (30% of total score)
            3. Project and Achievement Alignment (20% of total score)
            4. Education and Certifications (10% of total score)
            5. Qualitative Assessment
            6. Recommendations

            Ensure all scores are calculated accurately and provide detailed explanations for each section.
            """,
            input_variables=["job_description", "candidate_resume"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser
        
        # Initialize cache with JSONDisk
        #cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resume_job_match_cache')
        #self.cache = Cache(directory=cache_dir, disk=JSONDisk, disk_compress_level=6)

    def generate_match_analysis(self, job_description: str, candidate_resume: str) -> ResumeJobMatch:
        # Create a unique key for caching
        #cache_key = self._create_cache_key(job_description, candidate_resume)
        
        # Try to get the result from cache
        #cached_result = self.cache.get(cache_key)
        #if cached_result:
        #    logger.info("Retrieved result from cache")
        #    return ResumeJobMatch(**cached_result)
        
        # If not in cache, generate the analysis
        logger.info("Generating new resume-job match analysis using API")
        analysis = self.chain.invoke({
            "job_description": job_description,
            "candidate_resume": candidate_resume
        })
        
        # Cache the result
        #self.cache.set(cache_key, analysis)
        
        return analysis

    def _create_cache_key(self, job_description: str, candidate_resume: str) -> str:
        # Create a unique key based on input parameters
        key_data = f"{job_description}|{candidate_resume}"
        return zlib.adler32(key_data.encode())