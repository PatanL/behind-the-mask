// The home page: the Steer AI exhibit's live android (web/src/main.js, unchanged), in the launchpad's style, with the
// site's nav and the wallet. Live steering by default; ready-made answers are the other mode.
import '../../src/style.css';
import './landing.css';
import '../../src/main.js';
import { pick, onWallet } from './wallet.js';
const b = document.getElementById('wallet-top');
onWallet((a) => { b.textContent = a ? `${a.slice(0, 4)}…${a.slice(-4)}` : 'Connect wallet'; b.classList.toggle('on', !!a); });
b.onclick = () => pick().catch(() => {});   // connected: the picker shows the wallet, copy, disconnect, switch
