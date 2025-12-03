interface LogoProps {
	className?: string;
}

export function Logo({ className = "" }: LogoProps) {
	return (
		<svg
			viewBox="0 0 32 32"
			fill="none"
			xmlns="http://www.w3.org/2000/svg"
			className={className}
		>
			{/* Three concentric arcs suggesting fine-tuning levels */}
			<path
				d="M 16 4 A 12 12 0 0 1 28 16"
				stroke="#9CA3AF"
				strokeWidth="2.5"
				strokeLinecap="round"
				opacity="0.4"
			/>
			<path
				d="M 28 16 A 12 12 0 0 1 16 28"
				stroke="#6B7280"
				strokeWidth="2.5"
				strokeLinecap="round"
				opacity="0.6"
			/>
			<path
				d="M 16 28 A 12 12 0 0 1 4 16"
				stroke="#4B5563"
				strokeWidth="2.5"
				strokeLinecap="round"
				opacity="0.8"
			/>

			{/* Central adjustment nodes in a triangle formation */}
			<circle cx="16" cy="12" r="2" fill="#4B5563" />
			<circle cx="20" cy="20" r="2" fill="#6B7280" />
			<circle cx="12" cy="20" r="2" fill="#6B7280" />

			{/* Connecting lines suggesting automation flow */}
			<line
				x1="16" y1="12"
				x2="20" y2="20"
				stroke="#9CA3AF"
				strokeWidth="1.5"
				opacity="0.5"
			/>
			<line
				x1="20" y1="20"
				x2="12" y2="20"
				stroke="#9CA3AF"
				strokeWidth="1.5"
				opacity="0.5"
			/>
			<line
				x1="12" y1="20"
				x2="16" y2="12"
				stroke="#9CA3AF"
				strokeWidth="1.5"
				opacity="0.5"
			/>
		</svg>
	);
}
