// Minimal Cloudflare Worker bridge. Keep RUNPOD_API_KEY secret.
export interface Env { RUNPOD_API_KEY:string; RUNPOD_ENDPOINT_ID:string; }
export default { async fetch(req:Request, env:Env):Promise<Response> {
  if(req.method!=="POST") return new Response("POST only",{status:405});
  const body:any=await req.json();
  const r=await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/runsync`,{
    method:"POST",
    headers:{Authorization:`Bearer ${env.RUNPOD_API_KEY}`,"Content-Type":"application/json"},
    body:JSON.stringify({input:{action:"generate",text:String(body.text||""),cfg_value:2.0,inference_timesteps:10}})
  });
  return new Response(await r.text(),{status:r.status,headers:{"Content-Type":"application/json"}});
}};
