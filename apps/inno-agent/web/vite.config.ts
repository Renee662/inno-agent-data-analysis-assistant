import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { cpSync, existsSync, symlinkSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const monoRoot = resolve(__dirname, "../../..");

// pi-web-ui depends on @lmstudio/sdk for model discovery,
// but inno-agent does not use LM Studio — stub it out to avoid bundling.
const stubLmStudioPlugin = {
	name: "stub-lmstudio-sdk",
	enforce: "pre" as const,
	resolveId(id: string) {
		if (id === "@lmstudio/sdk") return "\0stub:@lmstudio/sdk";
	},
	load(id: string) {
		if (id === "\0stub:@lmstudio/sdk") return "export const LMStudioClient = class {};";
	},
};

export default defineConfig({
	plugins: [
		stubLmStudioPlugin,
		react(),
		{
			name: "link-katex-fonts",
			buildStart() {
				// pi-web-ui's built CSS references url(fonts/KaTeX_...) relative to its dist/.
				// The actual fonts live in node_modules/katex/dist/fonts/.
				// Link the font directory so Vite can resolve it; copy as a fallback
				// when the host filesystem refuses symlink creation.
				const source = resolve(monoRoot, "node_modules/katex/dist/fonts");
				const target = resolve(monoRoot, "node_modules/@earendil-works/pi-web-ui/dist/fonts");
				if (!existsSync(target)) {
					try {
						symlinkSync(source, target, process.platform === "win32" ? "junction" : "dir");
					} catch (err) {
						if ((err as NodeJS.ErrnoException).code !== "EPERM") throw err;
						cpSync(source, target, { recursive: true });
					}
				}
			},
		},
		tailwindcss(),
	],
	server: {
		port: 5173,
		proxy: {
			"/api": {
				target: "http://localhost:3000",
				changeOrigin: true,
				ws: true,
			},
			"/health": "http://localhost:3000",
		},
	},
	build: {
		rollupOptions: {
			output: {
				manualChunks: {
					codemirror: [
						"@uiw/react-codemirror",
						"@codemirror/lang-cpp",
						"@codemirror/lang-css",
						"@codemirror/lang-go",
						"@codemirror/lang-html",
						"@codemirror/lang-java",
						"@codemirror/lang-javascript",
						"@codemirror/lang-json",
						"@codemirror/lang-markdown",
						"@codemirror/lang-python",
						"@codemirror/lang-rust",
						"@codemirror/lang-sql",
						"@codemirror/lang-xml",
						"@codemirror/lang-yaml",
					],
					"markdown-editor": ["@uiw/react-md-editor"],
					cytoscape: ["cytoscape", "cytoscape-cola", "cytoscape-cose-bilkent"],
					katex: ["katex"],
					"docx-preview": ["docx-preview"],
					xlsx: ["xlsx"],
				},
			},
		},
	},
});
