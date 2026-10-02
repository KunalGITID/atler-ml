// Runs ATLER's own TypeScript core (not a re-implementation) on benchmark
// jobs: JSON in on stdin, JSON out on stdout. ATLER_DIR points at a checkout
// of KunalGITID/ATLER. Node 23+ runs .ts files directly.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const dir = resolve(process.env.ATLER_DIR ?? '../ATLER');
const load = (p: string) => import(pathToFileURL(resolve(dir, 'src/core', p)).href);
// Subscription finding moved from import/statement.ts to import/recurring.ts
// when it got a model; older ATLER commits still have it in statement.ts.
const has = (p: string) => existsSync(resolve(dir, 'src/core', p));
const [classify, statement, sms, insights, recurring] = await Promise.all([
  load('classify.ts'), load('import/statement.ts'), load('import/sms.ts'), load('insights.ts'),
  load(has('import/recurring.ts') ? 'import/recurring.ts' : 'import/statement.ts'),
]);

const suggest = await load('suggest.ts');
const fb = has('alertFeedback.ts') ? await load('alertFeedback.ts') : null;
const categoryPrior = has('categoryPrior.ts') ? await load('categoryPrior.ts') : null;
const loadPrior = () => JSON.parse(readFileSync(resolve(dir, 'src/core/categoryPrior.json'), 'utf8'));

type Job =
  | { task: 'classify'; train: { name: string; categoryId: string }[]; test: string[] }
  | { task: 'keywords'; names: string[] }
  | { task: 'recurring'; debits: { on: string; description: string; amount: number }[]; today: string }
  | { task: 'candidates'; debits: { on: string; description: string; amount: number }[]; today: string }
  | { task: 'suggest'; categories: string[]; filed: { name: string; category: string }[]; test: string[]; prior: boolean }
  | { task: 'priorProbabilities'; texts: string[] }
  | { task: 'feedback'; payments: { id: string; name: string; amount: number; on: string; categoryId: string | null; anomaly: boolean }[]; answerRate: number; learn: boolean; reviewFrom?: string }
  | { task: 'features' }
  | { task: 'treeProbability'; x: number[][] }
  | { task: 'unusual'; payments: { id: string; name: string; amount: number; on: string; categoryId: string | null }[] };

function run(job: Job): unknown {
  switch (job.task) {
    case 'classify': {
      const model = classify.train(job.train);
      return job.test.map(name => classify.predict(model, name));
    }
    case 'keywords':
      return job.names.map(n => sms.suggestCategory(n));
    case 'recurring':
      return {
        found: recurring.findRecurring(job.debits, job.today),
        merchants: job.debits.map(d => statement.merchantName(d.description)),
      };
    case 'candidates': {
      // Each debit carries its row number through, so Python can label the series.
      const debits = job.debits.map((d, i) => ({ ...d, i }));
      return recurring.recurringCandidates(debits).map((c: { name: string; whole: boolean; series: { i: number }[] }) => ({
        name: c.name, whole: c.whole, rows: c.series.map(d => d.i), features: recurring.seriesFeatures(c.series, job.today),
      }));
    }
    case 'suggest': {
      // ATLER's real suggestCategoryId, with your categories and what you've filed so far.
      const categories = job.categories.map(name => ({ id: name, name, budget: null }));
      const payments = job.filed.map((f, i) => ({ id: `p${i}`, name: f.name, amount: 100, on: '2026-01-01', categoryId: f.category, source: 'statement' }));
      const prior = job.prior ? loadPrior() : null;
      return job.test.map(name => suggest.suggestCategoryId(name, categories, payments, [], prior));
    }
    case 'priorProbabilities': {
      const prior = loadPrior();
      return job.texts.map(t => { const p = categoryPrior.priorProbabilities(prior, t); return p ? prior.classes.map((c: string) => p.get(c)) : null; });
    }
    case 'feedback': {
      // A user who answers a share of the alerts they're shown, truthfully.
      // Payments in date order; each is judged with the bars learned so far.
      let seed = 12345;
      const random = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
      const payments = job.payments.map(p => ({ ...p, source: 'statement' }));
      const verdicts: unknown[] = [];
      let bars = fb.learnBars([]);
      // reviewFrom: everything before it is an imported statement. On that day
      // you review its biggest past jumps (insights.pastJumps), then go live.
      let reviewed = !job.reviewFrom;
      return payments.map((p, i) => {
        if (job.reviewFrom && p.on < job.reviewFrom) return null;
        if (!reviewed) {
          reviewed = true;
          const known = payments.slice(0, i);
          for (const u of insights.pastJumps(known, p.on, verdicts)) {
            verdicts.push({ paymentId: u.payment.id, merchant: insights.merchantOf(u.payment), times: u.times, expected: !u.payment.anomaly, at: i });
          }
          bars = fb.learnBars(verdicts);
        }
        const u = insights.unusualness(p, payments.slice(0, i + 1), job.learn ? bars : undefined);
        if (!u) return null;
        if (random() < job.answerRate) {
          verdicts.push({ paymentId: p.id, merchant: insights.merchantOf(p), times: u.times, expected: !p.anomaly, at: i });
          bars = fb.learnBars(verdicts);
        }
        return u.times;
      });
    }
    case 'features':
      return recurring.SERIES_FEATURES;
    case 'treeProbability': {
      const model = JSON.parse(readFileSync(resolve(dir, 'src/core/import/subscriptionModel.json'), 'utf8'));
      return job.x.map(x => recurring.treeProbability(model, x));
    }
    case 'unusual': {
      const payments = job.payments.map(p => ({ ...p, source: 'statement' }));
      return payments.map(p => insights.unusualness(p, payments)?.times ?? null);
    }
  }
}

const jobs: Job[] = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(jobs.map(run)));
