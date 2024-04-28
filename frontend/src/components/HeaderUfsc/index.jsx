import { Logo } from "./Logo";
import { Topbar } from "./Topbar";
import { Shortcuts } from "./Shortcuts";

export function HeaderUfsc() {
    return(
        <>
            <header>
                <div>
                    <div>
                        <Shortcuts/>
                        <Topbar/>
                        <Logo/>
                    </div>
                </div>
            </header>
        </>
    );
}