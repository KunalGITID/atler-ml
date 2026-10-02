// Runs ATLER's own TypeScript core (not a re-implementation) on benchmark
// jobs: JSON in on stdin, JSON out on stdout. ATLER_DIR points at a checkout
// of KunalGITID/ATLER. Node 23+ runs .ts files directly.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const dir = resolve(process.env.ATLER_DIR ?? '../ATLER');
const load = (p: string) => import(pathToFileURL(resolve(dir, 'src/core', p)).href);
const [classify, statement, sms, insights] = await Promise.all([
  load('classify.ts'), load('import/statement.ts'), load('import/sms.ts'), load('insights.ts'),
]);

type Job =
  | { task: 'classify'; train: { name: string; categoryId: string }[]; test: string[] }
  | { task: 'keywords'; names: string[] }
  | { task: 'recurring'; debits: { on: string; description: string; amount: number }[]; today: string }
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
        found: statement.findRecurring(job.debits, job.today),
        merchants: job.debits.map(d => statement.merchantName(d.description)),
      };
    case 'unusual': {
      const payments = job.payments.map(p => ({ ...p, source: 'statement' }));
      return payments.map(p => insights.unusualness(p, payments)?.times ?? null);
    }
  }
}

const jobs: Job[] = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(jobs.map(run)));
