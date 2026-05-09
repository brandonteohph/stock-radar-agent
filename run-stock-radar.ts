import {Agent, Runner, webSearchTool, withTrace} from "@openai/agents";
import {z} from "zod";
import {resolve} from "node:path";
import {fileURLToPath} from "node:url";


// Tool definitions
const webSearchPreview = webSearchTool({
    searchContextSize: "medium",
    userLocation: {
        type: "approximate"
    }
})
const NewsSearchAgentSchema = z.object({
    ticker: z.string(),
    company_name: z.string(),
    articles: z.array(z.object({
        title: z.string(),
        url: z.string(),
        source: z.string(),
        published_date: z.string(),
        summary: z.string()
    }))
});
const BriefWriterAgentSchema = z.object({
    ticker: z.string(),
    date: z.string(),
    headline_summary: z.string(),
    overall_sentiment: z.enum(["positive", "negative", "mixed", "neutral"]),
    top_links: z.array(z.object({title: z.string(), url: z.string(), why_it_matters: z.string()})),
    watch_items: z.array(z.string()),
    uncertainty: z.array(z.string()),
    disclaimer: z.string()
});
const NewsTriageAgentSchema = z.object({
    ticker: z.string(),
    sentiment: z.enum(["positive", "negative", "mixed", "neutral"]),
    ranked_articles: z.array(z.object({
        rank: z.number().int(),
        title: z.string(),
        url: z.string(),
        source: z.string(),
        topic: z.enum(["earnings", "guidance", "product", "regulation", "macro", "analyst_rating", "leadership", "legal", "m_and_a", "other"]),
        sentiment: z.enum(["positive", "negative", "mixed", "neutral"]),
        why_it_matters: z.string()
    })),
    key_themes: z.array(z.string())
});
const newsSearchAgent = new Agent({
    name: "News Search Agent",
    instructions: `    Find recent, relevant news links for the ticker/company.
    Prefer primary sources, SEC/company investor relations, Reuters/AP/CNBC/Bloomberg/Yahoo Finance/WSJ/FT/MarketWatch when available.
    Avoid duplicate syndicated versions of the same story.`,
    model: "gpt-5-nano",
    tools: [
        webSearchPreview
    ],
    outputType: NewsSearchAgentSchema,
    modelSettings: {
        reasoning: {
            effort: "low",
            summary: "auto"
        },
        store: true
    }
});

const briefWriterAgent = new Agent({
    name: "Brief Writer Agent",
    instructions: `    Produce a concise review brief.
    Include links.
    Do not make investment recommendations.`,
    model: "gpt-5-nano",
    outputType: BriefWriterAgentSchema,
    modelSettings: {
        reasoning: {
            effort: "minimal",
            summary: "auto"
        },
        store: true
    }
});

const newsTriageAgent = new Agent({
    name: "News Triage Agent",
    instructions: `    Rank the links by relevance to investors.
    Remove low-signal articles.
    Classify each item by topic:
      earnings, guidance, product, regulation, macro, analyst rating, leadership, legal, M&A, other.
`,
    model: "gpt-5-nano",
    outputType: NewsTriageAgentSchema,
    modelSettings: {
        reasoning: {
            effort: "low",
            summary: "auto"
        },
        store: true
    }
});

// Main code entrypoint
type WorkflowInput = {
    ticker: string;
    company_name: string;
    max_links?: number;
};

export const runWorkflow = async (workflow: WorkflowInput) => {
    return await withTrace("Stock News Agent", async () => {
        const ticker = workflow.ticker;
        const companyName = workflow.company_name;
        const maxLinks = workflow.max_links ?? 5;

        const runner = new Runner({
            traceMetadata: {
                __trace_source__: "agent-builder",
                workflow_id: "wf_69fcb66f25ac8190ba0c24f7fe0859ae07c686e83a6570b4"
            }
        });

        const newsSearchAgentResultTemp = await runner.run(newsSearchAgent, [
            {
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text: JSON.stringify({
                            ticker,
                            company_name: companyName,
                            max_links: maxLinks,
                            task: `Find up to ${maxLinks} recent investor-relevant news articles for ${companyName} (${ticker}).`
                        })
                    }
                ]
            }
        ]);

        if (!newsSearchAgentResultTemp.finalOutput) {
            throw new Error("News Search Agent result is undefined");
        }

        const newsSearchOutput = newsSearchAgentResultTemp.finalOutput;

        const newsTriageAgentResultTemp = await runner.run(newsTriageAgent, [
            {
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text: JSON.stringify({
                            ticker,
                            company_name: companyName,
                            articles: newsSearchOutput.articles,
                            task: "Rank these articles by investor relevance, remove duplicates, classify topic and sentiment, and identify key themes."
                        })
                    }
                ]
            }
        ]);

        if (!newsTriageAgentResultTemp.finalOutput) {
            throw new Error("News Triage Agent result is undefined");
        }

        const newsTriageOutput = newsTriageAgentResultTemp.finalOutput;

        const briefWriterAgentResultTemp = await runner.run(briefWriterAgent, [
            {
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text: JSON.stringify({
                            ticker,
                            company_name: companyName,
                            sentiment: newsTriageOutput.sentiment,
                            ranked_articles: newsTriageOutput.ranked_articles,
                            key_themes: newsTriageOutput.key_themes,
                            task: "Write the final daily stock radar brief using only this triaged data."
                        })
                    }
                ]
            }
        ]);

        if (!briefWriterAgentResultTemp.finalOutput) {
            throw new Error("Brief Writer Agent result is undefined");
        }

        return briefWriterAgentResultTemp.finalOutput;
    });
}

type Brief = z.infer<typeof BriefWriterAgentSchema>;

const requiredEnv = (name: string): string => {
    const value = process.env[name];
    if (!value) {
        throw new Error(`Missing required environment variable: ${name}`);
    }

    return value;
};

const parseArgs = (args: string[]): Partial<WorkflowInput> => {
    const parsed: Partial<WorkflowInput> = {};

    for (let index = 0; index < args.length; index += 1) {
        const arg = args[index];
        const nextValue = args[index + 1];

        if (arg === "--ticker" && nextValue) {
            parsed.ticker = nextValue;
            index += 1;
        } else if (arg === "--company-name" && nextValue) {
            parsed.company_name = nextValue;
            index += 1;
        } else if (arg === "--max-links" && nextValue) {
            parsed.max_links = Number.parseInt(nextValue, 10);
            index += 1;
        }
    }

    return parsed;
};

const formatBriefMarkdown = (brief: Brief): string => {
    const links = brief.top_links
        .map((link, index) => `${index + 1}. [${link.title}](${link.url}) - ${link.why_it_matters}`)
        .join("\n");
    const watchItems = brief.watch_items.map((item) => `- ${item}`).join("\n");
    const uncertainty = brief.uncertainty.map((item) => `- ${item}`).join("\n");

    return `# Daily Stock Radar: ${brief.ticker}

Date: ${brief.date}
Sentiment: ${brief.overall_sentiment}

${brief.headline_summary}

## Top Links
${links || "- No links returned."}

## Watch Items
${watchItems || "- No watch items returned."}

## Uncertainty
${uncertainty || "- No uncertainty items returned."}

${brief.disclaimer}
`;
};

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1]);

if (isMain) {
    const args = parseArgs(process.argv.slice(2));
    const ticker = args.ticker ?? process.env.TICKER ?? "NVDA";
    const companyName = args.company_name ?? process.env.COMPANY_NAME ?? ticker;
    const maxLinks = args.max_links ?? Number.parseInt(process.env.MAX_LINKS ?? "5", 10);

    requiredEnv("OPENAI_API_KEY");

    if (!Number.isFinite(maxLinks) || maxLinks < 1) {
        throw new Error("MAX_LINKS must be a positive integer");
    }

    const brief = await runWorkflow({
        ticker,
        company_name: companyName,
        max_links: maxLinks
    });

    console.log(formatBriefMarkdown(brief));
}
