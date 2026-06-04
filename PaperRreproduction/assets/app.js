function copyCode(id){const el=document.getElementById(id); if(!el){return} navigator.clipboard.writeText(el.innerText).then(()=>alert('程式碼已複製'));}
window.addEventListener('load',()=>{document.querySelectorAll('.step').forEach((s,i)=>s.style.animationDelay=(i*0.08)+'s');});

window.addEventListener('load',()=>{
  document.querySelectorAll('.bar-fill').forEach((b,i)=>b.style.animationDelay=(i*0.08)+'s');
  document.querySelectorAll('details summary').forEach(s=>s.setAttribute('title','點擊展開 / 收合'));
});
