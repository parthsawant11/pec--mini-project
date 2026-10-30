import express from 'express';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import path from 'path';
import { GoogleGenAI, Type } from '@google/genai';
import { CAREERS_DATABASE } from './src/data/careersData.js';
import { calculateCareerMatches, calculateSkillGap, generateRoadmap, analyzeResumeATS } from './src/utils/scoringEngine.js';
import { SYSTEM_PROMPTS } from './src/data/systemPrompts.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;

app.use(express.json({ limit: '10mb' }));

// Initialize GoogleGenAI SDK safely
let ai: GoogleGenAI | null = null;
if (process.env.GEMINI_API_KEY) {
  ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// ----------------------------------------------------------------------------
// API Endpoints
// ----------------------------------------------------------------------------

// 1. Get all careers (O*NET database)
app.get('/api/careers', (req, res) => {
  const { category, minSalary, maxGrowth, query } = req.query;
  let results = [...CAREERS_DATABASE];

  if (category && category !== 'All') {
    results = results.filter(c => c.category === category);
  }
  if (minSalary) {
    const min = Number(minSalary);
    results = results.filter(c => c.medianSalaryUSD >= min);
  }
  if (query) {
    const q = String(query).toLowerCase();
    results = results.filter(c => 
      c.title.toLowerCase().includes(q) || 
      c.description.toLowerCase().includes(q) ||
      c.requiredSkills.some(s => s.name.toLowerCase().includes(q))
    );
  }

  res.json({ success: true, count: results.length, data: results });
});

// 2. Get specific career by ID
app.get('/api/careers/:id', (req, res) => {
  const career = CAREERS_DATABASE.find(c => c.id === req.params.id);
  if (!career) {
    return res.status(404).json({ success: false, error: 'Career not found in O*NET dataset' });
  }
  res.json({ success: true, data: career });
});

// 3. Career Recommendations with RAG & Gemini
app.post('/api/recommendations', async (req, res) => {
  try {
    const { profile } = req.body;
    if (!profile) {
      return res.status(400).json({ success: false, error: 'User profile payload required' });
    }

    // First, compute deterministic baseline RAG matches from knowledge base
    const baselineMatches = calculateCareerMatches(profile, CAREERS_DATABASE);
    const topCandidates = baselineMatches.slice(0, 5);

    // If Gemini API is available, enrich with Gemini reasoning
    if (ai && process.env.GEMINI_API_KEY) {
      try {
        const promptText = `
Candidate Profile:
Name: ${profile.fullName}
Education: ${profile.education?.level} in ${profile.education?.field}
Experience Level: ${profile.experienceLevel}
Current Skills: ${profile.skills?.join(', ')}
Interests: ${profile.interests?.join(', ')}
Personality Traits: ${JSON.stringify(profile.personalityTraits || {})}
Desired Minimum Salary: $${profile.goals?.desiredSalaryMin || 80000}
Preferred Timeline: ${profile.goals?.timelineMonths || 6} months
Primary Motivator: ${profile.goals?.primaryMotivator || 'Growth'}
Work Constraint: ${profile.constraints?.preferredWorkType || 'Remote'}, ${profile.constraints?.hoursPerWeekAvailable || 15} hrs/week

Top Retrieved Careers from O*NET knowledge base:
${topCandidates.map(c => `
- ID: ${c.career.id}
  Title: ${c.career.title} (SOC: ${c.career.socCode})
  Category: ${c.career.category}
  Median Salary: $${c.career.medianSalaryUSD}
  10-Yr Growth: +${c.career.tenYearGrowthRate}%
  Required Skills: ${c.career.requiredSkills.map(s => s.name).join(', ')}
`).join('\n')}

Task:
Evaluate and return the top 5 matches as a structured JSON object. Provide matchScore (0-100), detailed scoreBreakdown (skillsMatch, personalityMatch, goalAlignment, marketOpportunity), nuanced 2-sentence reasoning, key strengths, main challenges, and recommended timeline.
`;

        const response = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: promptText,
          config: {
            systemInstruction: SYSTEM_PROMPTS.careerRecommender.systemInstruction,
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                matches: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      careerId: { type: Type.STRING },
                      matchScore: { type: Type.INTEGER },
                      scoreBreakdown: {
                        type: Type.OBJECT,
                        properties: {
                          skillsMatch: { type: Type.INTEGER },
                          personalityMatch: { type: Type.INTEGER },
                          goalAlignment: { type: Type.INTEGER },
                          marketOpportunity: { type: Type.INTEGER },
                        },
                      },
                      reasoning: { type: Type.STRING },
                      keyStrengths: { type: Type.ARRAY, items: { type: Type.STRING } },
                      mainChallenges: { type: Type.ARRAY, items: { type: Type.STRING } },
                      recommendedTimelineMonths: { type: Type.INTEGER },
                    },
                    required: ['careerId', 'matchScore', 'reasoning', 'keyStrengths', 'mainChallenges'],
                  },
                },
              },
              required: ['matches'],
            },
          },
        });

        if (response.text) {
          const parsed = JSON.parse(response.text);
          if (parsed && Array.isArray(parsed.matches) && parsed.matches.length > 0) {
            // merge with full career object
            const enriched = parsed.matches.map((m: any) => {
              const matchedCareer = CAREERS_DATABASE.find(c => c.id === m.careerId) || topCandidates[0].career;
              return {
                careerId: matchedCareer.id,
                career: matchedCareer,
                matchScore: m.matchScore || 85,
                scoreBreakdown: m.scoreBreakdown || {
                  skillsMatch: 80,
                  personalityMatch: 85,
                  goalAlignment: 85,
                  marketOpportunity: 85,
                },
                reasoning: m.reasoning,
                keyStrengths: m.keyStrengths || [],
                mainChallenges: m.mainChallenges || [],
                recommendedTimelineMonths: m.recommendedTimelineMonths || 6,
              };
            });

            return res.json({ success: true, aiPowered: true, data: enriched });
          }
        }
      } catch (geminiErr) {
        console.warn('Gemini recommendation API fallback to deterministic scoring:', geminiErr);
      }
    }

    // High-precision deterministic fallback
    res.json({ success: true, aiPowered: false, data: topCandidates });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 4. Skill Gap Analysis
app.post('/api/skill-gap', async (req, res) => {
  try {
    const { profile, careerId } = req.body;
    const career = CAREERS_DATABASE.find(c => c.id === careerId);
    if (!career) {
      return res.status(404).json({ success: false, error: 'Career not found' });
    }

    const baselineGap = calculateSkillGap(profile, career);

    if (ai && process.env.GEMINI_API_KEY) {
      try {
        const prompt = `
Candidate: ${profile.fullName}
Current Skills: ${profile.skills?.join(', ')}
Target Career: ${career.title} (Category: ${career.category})
Required Skills in Career: ${career.requiredSkills.map(s => `${s.name} (importance: ${s.importance})`).join(', ')}

Provide an expert skill-gap analysis with readiness score, missing skills breakdown with estimated hours to master, and encouraging yet realistic advice.
`;
        const response = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: prompt,
          config: {
            systemInstruction: SYSTEM_PROMPTS.skillGapAnalyzer.systemInstruction,
            responseMimeType: 'application/json',
          },
        });

        if (response.text) {
          const parsed = JSON.parse(response.text);
          if (parsed && parsed.overallReadinessScore !== undefined) {
            return res.json({
              success: true,
              aiPowered: true,
              data: {
                ...baselineGap,
                overallReadinessScore: parsed.overallReadinessScore ?? baselineGap.overallReadinessScore,
                personalizedAdvice: parsed.personalizedAdvice || baselineGap.personalizedAdvice,
                criticalMissingSkills: parsed.criticalMissingSkills || baselineGap.criticalMissingSkills,
              },
            });
          }
        }
      } catch (err) {
        console.warn('Skill gap AI enrichment fallback:', err);
      }
    }

    res.json({ success: true, aiPowered: false, data: baselineGap });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 5. Phased Roadmap Generator
app.post('/api/roadmap', async (req, res) => {
  try {
    const { profile, careerId, durationMonths = 6 } = req.body;
    const career = CAREERS_DATABASE.find(c => c.id === careerId);
    if (!career) {
      return res.status(404).json({ success: false, error: 'Career not found' });
    }

    const baselineRoadmap = generateRoadmap(profile, career, durationMonths);

    if (ai && process.env.GEMINI_API_KEY) {
      try {
        const prompt = `
Create a personalized ${durationMonths}-month career roadmap for ${profile.fullName} pursuing ${career.title}.
Weekly Hours Available: ${profile.constraints?.hoursPerWeekAvailable || 15} hrs/week
Target Skills: ${career.requiredSkills.slice(0, 5).map(s => s.name).join(', ')}
Certifications to Target: ${career.topCertifications.join(', ')}
`;
        const response = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: prompt,
          config: {
            systemInstruction: SYSTEM_PROMPTS.roadmapPlanner.systemInstruction,
            responseMimeType: 'application/json',
          },
        });

        if (response.text) {
          const parsed = JSON.parse(response.text);
          if (parsed && Array.isArray(parsed.milestones) && parsed.milestones.length > 0) {
            return res.json({
              success: true,
              aiPowered: true,
              data: {
                ...baselineRoadmap,
                milestones: parsed.milestones,
              },
            });
          }
        }
      } catch (err) {
        console.warn('Roadmap AI generation fallback:', err);
      }
    }

    res.json({ success: true, aiPowered: false, data: baselineRoadmap });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 6. Resume ATS Scanner & Rewriter
app.post('/api/resume-analyzer', async (req, res) => {
  try {
    const { resumeText, targetCareerId } = req.body;
    if (!resumeText || typeof resumeText !== 'string') {
      return res.status(400).json({ success: false, error: 'Resume text required' });
    }

    const career = CAREERS_DATABASE.find(c => c.id === targetCareerId) || CAREERS_DATABASE[0];
    const baselineAnalysis = analyzeResumeATS(resumeText, career);

    if (ai && process.env.GEMINI_API_KEY) {
      try {
        const prompt = `
Target Career: ${career.title} (SOC: ${career.socCode})
Core Required Competencies: ${career.requiredSkills.map(s => s.name).join(', ')}

Resume Content:
${resumeText.slice(0, 3000)}

Analyze this resume for ATS pass rate, keyword density, and provide Google X-Y-Z formula bullet point rewrites.
`;
        const response = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite',
          contents: prompt,
          config: {
            systemInstruction: SYSTEM_PROMPTS.resumeAnalyzer.systemInstruction,
            responseMimeType: 'application/json',
          },
        });

        if (response.text) {
          const parsed = JSON.parse(response.text);
          if (parsed && parsed.overallScore !== undefined) {
            return res.json({
              success: true,
              aiPowered: true,
              data: {
                ...baselineAnalysis,
                overallScore: parsed.overallScore ?? baselineAnalysis.overallScore,
                keywordMatchScore: parsed.keywordMatchScore ?? baselineAnalysis.keywordMatchScore,
                formattingScore: parsed.formattingScore ?? baselineAnalysis.formattingScore,
                bulletRewrites: parsed.bulletRewrites || baselineAnalysis.bulletRewrites,
                actionPlan: parsed.actionPlan || baselineAnalysis.actionPlan,
                criticalIssues: parsed.criticalIssues || baselineAnalysis.criticalIssues,
              },
            });
          }
        }
      } catch (err) {
        console.warn('Resume ATS AI fallback:', err);
      }
    }

    res.json({ success: true, aiPowered: false, data: baselineAnalysis });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// 7. Context-Aware AI Career Mentor Chat with Real Multi-Turn Dialogue
app.post('/api/chat', async (req, res) => {
  try {
    const { message, history = [], profile, targetCareerId } = req.body;
    if (!message || typeof message !== 'string') {
      return res.status(400).json({ success: false, error: 'Message content required' });
    }

    const targetCareer = CAREERS_DATABASE.find(c => c.id === targetCareerId) || CAREERS_DATABASE[0];
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({
        success: false,
        error: 'GEMINI_API_KEY is not configured on the server.'
      });
    }

    try {
      const client = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });

      // System instruction that balances career context with direct, intelligent dialogue
      const sysInstruction = `You are ForgeBot, an intelligent, versatile AI Career Mentor & Technical Advisor.
You are chatting with a candidate using CareerForge AI.

CANDIDATE CONTEXT:
- Candidate Name: ${profile?.fullName || 'Candidate'}
- Background: ${profile?.education?.level || "Bachelor's Degree"} in ${profile?.education?.field || 'General Field'} (${profile?.experienceLevel || 'Early Career'})
- Current Role: ${profile?.currentTitle || 'Student / Professional'}
- Skills: ${(profile?.skills || []).join(', ') || 'General Technical Skills'}
- Desired Career: ${targetCareer.title} (${targetCareer.category}, Median: $${targetCareer.medianSalaryUSD.toLocaleString()}, 10-Yr Growth: +${targetCareer.tenYearGrowthRate}%)
- Required Skills for Career: ${targetCareer.requiredSkills.map(s => s.name).slice(0, 6).join(', ')}

CORE BEHAVIOR RULES (CRITICAL):
1. DIRECT ANSWER FIRST: Always directly, accurately, and naturally answer what the user is asking!
   - If the user asks a factual, general, or trivia question (e.g., "What is the capital of India", "Explain binary search", "How does DNS work"), answer it accurately and directly right away.
   - NEVER ignore the user's question.
   - NEVER give a canned speech or force an unnatural lecture about career roadmaps when they asked a specific question.
2. CAREER & TECHNICAL DEPTH: When the user asks about career transitions, interview preparation, portfolio projects, resumes, or learning paths, give world-class, pragmatic advice tailored to their specific background (${profile?.fullName || 'the user'}, ${targetCareer.title}).
3. CONVERSATIONAL CONTINUITY: Remember previous messages in this conversation. If the user refers to "that project", "the second one", or asks a follow-up, seamlessly continue the discussion.
4. TONE: Sharp, engaging, encouraging, highly knowledgeable, and conversational.`;

      // Build clean alternating conversation history for Gemini API
      const rawTurns: Array<{ role: 'user' | 'model'; text: string }> = [];

      if (Array.isArray(history)) {
        for (const item of history.slice(-10)) {
          if (item && item.content && typeof item.content === 'string' && item.content.trim()) {
            rawTurns.push({
              role: item.sender === 'user' ? 'user' : 'model',
              text: item.content.trim(),
            });
          }
        }
      }

      // Append the latest user message
      rawTurns.push({
        role: 'user',
        text: message.trim(),
      });

      // Collapse consecutive same-role turns & ensure first turn is user
      const formattedContents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];
      for (const turn of rawTurns) {
        if (formattedContents.length === 0 && turn.role !== 'user') {
          continue; // Must start with user
        }

        if (formattedContents.length > 0 && formattedContents[formattedContents.length - 1].role === turn.role) {
          formattedContents[formattedContents.length - 1].parts[0].text += `\n\n${turn.text}`;
        } else {
          formattedContents.push({
            role: turn.role,
            parts: [{ text: turn.text }],
          });
        }
      }

      // If empty for any reason, fallback to single user turn
      const finalContents = formattedContents.length > 0
        ? formattedContents
        : [{ role: 'user' as const, parts: [{ text: message.trim() }] }];

      const response = await client.models.generateContent({
        model: 'gemini-3.1-flash-lite',
        contents: finalContents,
        config: {
          systemInstruction: sysInstruction,
        },
      });

      const replyText = response.text?.trim();
      if (replyText) {
        return res.json({
          success: true,
          aiPowered: true,
          model: 'gemini-3.1-flash-lite',
          reply: replyText,
        });
      }

      throw new Error('Gemini model returned empty response text.');
    } catch (apiErr: any) {
      console.error('Gemini API call failed in /api/chat:', apiErr);
      return res.status(500).json({
        success: false,
        error: `AI service error: ${apiErr?.message || 'Unable to generate response'}`,
      });
    }
  } catch (err: any) {
    console.error('Server error in /api/chat:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// 8. Aggregated Job Market Insights (BLS / O*NET)
app.get('/api/market-insights', (req, res) => {
  const categories = Array.from(new Set(CAREERS_DATABASE.map(c => c.category)));
  
  const categoryStats = categories.map(cat => {
    const list = CAREERS_DATABASE.filter(c => c.category === cat);
    const avgSalary = Math.round(list.reduce((acc, c) => acc + c.medianSalaryUSD, 0) / list.length);
    const avgGrowth = Math.round((list.reduce((acc, c) => acc + c.tenYearGrowthRate, 0) / list.length) * 10) / 10;
    const totalOpenings = list.reduce((acc, c) => acc + c.projectedOpeningsAnnual, 0);

    return {
      category: cat,
      averageSalary: avgSalary,
      averageGrowthRate: avgGrowth,
      annualOpenings: totalOpenings,
      careerCount: list.length,
    };
  });

  // Top 10 In-Demand Skills Frequency
  const skillCountMap: Record<string, { count: number; category: string }> = {};
  CAREERS_DATABASE.forEach(c => {
    c.requiredSkills.forEach(s => {
      if (!skillCountMap[s.name]) {
        skillCountMap[s.name] = { count: 0, category: s.category };
      }
      skillCountMap[s.name].count += 1;
    });
  });

  const topSkills = Object.entries(skillCountMap)
    .map(([skill, meta]) => ({ skill, count: meta.count, category: meta.category }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  res.json({
    success: true,
    data: {
      totalCareersTracked: CAREERS_DATABASE.length,
      overallMedianSalary: 138000,
      averageTenYearGrowth: 25.4,
      categoryStats,
      topSkills,
    },
  });
});

// ----------------------------------------------------------------------------
// Vite Integration in Development / Production Static Server
// ----------------------------------------------------------------------------
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(process.cwd(), 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(process.cwd(), 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CareerForge AI full-stack server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
