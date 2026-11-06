import type { APIRoute } from 'astro';
import { buildSearchIndex } from '../../lib/searchIndex';

export const GET: APIRoute = async () => {
  const index = buildSearchIndex();
  return new Response(JSON.stringify(index), {
    headers: {
      'Content-Type': 'application/json'
    }
  });
};
